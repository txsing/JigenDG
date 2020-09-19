import argparse

import os
os.environ['KMP_WARNINGS'] = '0'
import torch
#from IPython.core.debugger import set_trace
from torch import nn
from torch.nn import functional as F
from data import data_helper
# from IPython.core.debugger import set_trace
from data.data_helper import available_datasets, pacs_datasets, digits_datasets, vlcs_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
import numpy as np
import random

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=225, help="Image size")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.0, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.0, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int, help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int, help="If set, it will limit the number of testing samples")

    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=31, help="Number of classes")
    parser.add_argument("--jigsaw_n_classes", "-jc", type=int, default=31, help="Number of classes for the jigsaw task")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="caffenet")
    parser.add_argument("--jig_weight", type=float, default=0.1, help="Weight for the jigsaw puzzle")
    parser.add_argument("--ooo_weight", type=float, default=0, help="Weight for odd one out task")

    # Boolean Flag
    # 'default' is the value that the attribute gets when the argument is absent
    parser.add_argument("--tf_logger", type=str2bool, nargs='?', 
                        const=True, default=True,
                        help="If true will save tensorboard compatible logs")
    parser.add_argument("--train_all", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="If true, all network weights will be trained")
    parser.add_argument("--nesterov", type=str2bool, nargs='?',
                        const=True, default=False, 
                        help="Use nesterov")
    parser.add_argument("--classify_only_sane", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--TTA",  type=str2bool, nargs='?',
                        const=True, default=False, 
                        help="Activate test time data augmentation")
    
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=None, type=float, help="If set, will bias the training procedure to show more often the whole image")
    

    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    # nesterov 是一种梯度下降的方法
    
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU No.")
    parser.add_argument("--seed", default=-1,  type=int, help="seed")
    
    return parser.parse_args()


# def compute_losses(net_output, jig_l, class_l):
#     return F.cross_entropy(net_output[0], jig_l), F.cross_entropy(net_output[1], class_l)

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        model = model_factory.get_network(args.network)(jigsaw_classes=args.jigsaw_n_classes + 1, classes=args.n_classes)
        self.model = model.to(device)

        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
        self.target_test_loaders = data_helper.get_jigsaw_test_dataloaders(args, patches=model.is_patch_based())
        # Evaluate on Validation & Test datasets
        self.evaluation_loaders = {"val": self.val_loader, "test": self.target_test_loaders}

        print("Dataset size: train %d, val %d" % (len(self.source_loader.dataset), len(self.val_loader.dataset)))
        
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, 
                                                                 args.train_all, nesterov=args.nesterov
                                                                )
        self.jig_weight = args.jig_weight
        self.only_non_scrambled = args.classify_only_sane
        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for it, ((data, jig_l, class_l), d_idx) in enumerate(self.source_loader):
            data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), d_idx.to(self.device)

            self.optimizer.zero_grad()
            jigsaw_logit, class_logit = self.model(data)
            jigsaw_loss = criterion(jigsaw_logit, jig_l)

            if self.only_non_scrambled: # 只对正常图片进行物种分类
                if self.target_id is not None:
                    # 图片没有被打乱 && 图片的 domain 不是 target domain
                    #（因为我们不训练target domain，target domain的图片只用来 predict）
                    idx = (jig_l == 0) & (d_idx != self.target_id)
                    class_loss = criterion(class_logit[idx], class_l[idx])
                else:
                    class_loss = criterion(class_logit[jig_l == 0], class_l[jig_l == 0])

            elif self.target_id: # 对所有（包括打乱的）图片进行物种分类，target domain 只用于 predict
                class_loss = criterion(class_logit[d_idx != self.target_id], class_l[d_idx != self.target_id])
            else: # 对所有（包括打乱的）图片进行物种分类，target domain 只用于 predict
                class_loss = criterion(class_logit, class_l)
            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            loss = class_loss + jigsaw_loss * self.jig_weight  # + 0.1 * domain_loss

            loss.backward()
            self.optimizer.step()

            self.logger.log(it, len(self.source_loader),
                            {"jigsaw": jigsaw_loss.item(),
                             "class": class_loss.item()
                            },
                            {"jigsaw": torch.sum(jig_pred == jig_l.data).item(),
                             "class": torch.sum(cls_pred == class_l.data).item(),
                            },
                            data.shape[0])
            # 解除变量引用与实际值的指向关系
            del loss, class_loss, jigsaw_loss, jigsaw_logit, class_logit

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.evaluation_loaders.items():
                if phase == 'test':
                    belonged_dataset = data_helper.get_belonged_dataset(self.args.source[0])
                    target_domains = [item for item in belonged_dataset if item not in self.args.source]

                    acc_sum = 0.0
                    for didx in range(len(loader)):
                        dkey = phase + '-' + target_domains[didx]

                        test_loader = loader[didx]
                        test_total = len(test_loader.dataset)
                        jigsaw_correct, class_correct = self.do_test(test_loader)

                        jigsaw_acc = float(jigsaw_correct) / total
                        class_acc = float(class_correct) / test_total

                        self.logger.log_test(dkey, {"class": class_acc})
                        if dkey not in self.results.keys():
                            self.results[dkey] = torch.zeros(self.args.epochs)
                        self.results[dkey][self.current_epoch] = class_acc
                        acc_sum += class_acc
                    self.logger.log_test(phase, {"class": acc_sum / len(loader)})
                    self.results[phase][self.current_epoch] = acc_sum / len(loader)
                else:
                    total = len(loader.dataset)
                    if loader.dataset.isMulti():
                        jigsaw_correct, class_correct, single_acc = self.do_test_multi(loader)
                        print("Single vs multi: %g %g" % (float(single_acc) / total, float(class_correct) / total))
                    else:
                        jigsaw_correct, class_correct = self.do_test(loader)

                    jigsaw_acc = float(jigsaw_correct) / total
                    class_acc = float(class_correct) / total
                    self.logger.log_test(phase, {"jigsaw": jigsaw_acc, "class": class_acc})
                    self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):
        jigsaw_correct = 0
        class_correct = 0
        domain_correct = 0
        for it, ((data, jig_l, class_l), _) in enumerate(loader):
            data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
            jigsaw_logit, class_logit = self.model(data)
            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
            jigsaw_correct += torch.sum(jig_pred == jig_l.data)
        return jigsaw_correct, class_correct

    def do_test_multi(self, loader):
        jigsaw_correct = 0
        class_correct = 0
        single_correct = 0
        for it, ((data, jig_l, class_l), d_idx) in enumerate(loader):
            data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
            n_permutations = data.shape[1]
            class_logits = torch.zeros(n_permutations, data.shape[0], self.n_classes).to(self.device)
            for k in range(n_permutations):
                class_logits[k] = F.softmax(self.model(data[:, k])[1], dim=1)
            class_logits[0] *= 4 * n_permutations  # bias more the original image
            class_logit = class_logits.mean(0)
            _, cls_pred = class_logit.max(dim=1)
            jigsaw_logit, single_logit = self.model(data[:, 0])
            _, jig_pred = jigsaw_logit.max(dim=1)
            _, single_logit = single_logit.max(dim=1)
            single_correct += torch.sum(single_logit == class_l.data)
            class_correct += torch.sum(cls_pred == class_l.data)
            jigsaw_correct += torch.sum(jig_pred == jig_l.data[:, 0])
        return jigsaw_correct, class_correct, single_correct

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)  # , "domain", "lambda"
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch()
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx__val_best = val_res.argmax()
        idx_test_best = test_res.argmax()
        print("Best test acc: %g in epoch: %d" % (test_res.max(), idx_test_best+1))
        self.logger.save_best(test_res[idx_test_best].item(), test_res.max().item())
        return self.logger, self.model


def main():
    args = get_args()
    if args.seed > 0:
        print("Using seed: %d!" % args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
