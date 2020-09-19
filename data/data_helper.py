from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data import StandardDataset
from data.JigsawLoader import JigsawDataset, JigsawTestDataset, get_split_dataset_info, _dataset_info, JigsawTestDatasetMultiple
from data.concat_dataset import ConcatDataset

vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
office_datasets = ["amazon", "dslr", "webcam"]
office_home_datasets = ["Art", "Clipart", "Product", "RealWorld"]
digits_datasets = ["mnist", "mnist_m", "svhn", "synth", "usps"]
available_datasets = vlcs_datasets + pacs_datasets + office_datasets + office_home_datasets + digits_datasets + ["ALL"]


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def get_train_dataloader(args, patches):
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    train_datasets = []
    val_datasets = []
    img_transformer, tile_transformer = get_train_transformers(args)
    limit = args.limit_source
    for dname in dataset_list:
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(
            join(dirname(__file__), 'txt_lists','%s_train.txt' % dname),
            args.val_size
        )
        train_dataset = JigsawDataset(name_train, labels_train, patches=patches,
                                      img_transformer=img_transformer,
                                      tile_transformer=tile_transformer,
                                      jig_classes=args.jigsaw_n_classes,
                                      bias_whole_image=args.bias_whole_image)
        if limit:
            train_dataset = Subset(train_dataset, limit)
        train_datasets.append(train_dataset)

        # Validation test => subtracted from train split
        val_datasets.append(
            JigsawTestDataset(name_val, labels_val, img_transformer=get_val_transformer(args),
                              patches=patches, jig_classes=args.jigsaw_n_classes))

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return train_loader, val_loader


def get_jigsaw_test_dataloaders(args, patches=False):
    if args.target == "ALL": # For Single Source Domain Generalization
        belonged_dataset = get_belonged_dataset(args.source[0])
        target_domains = [item for item in belonged_dataset if item != args.source[0] ]
        
        test_loaders = []
        for dname in target_domains:
            test_loaders.append(get_single_test_dataloader(args, dname))
        return test_loaders
    else:
        return [get_single_test_dataloader(args, args.target)]

    
def get_single_test_dataloader(args, dname, patches=False):
    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', '%s_test.txt' % dname))
    img_tr = get_val_transformer(args)
    
    # JigsawTestDataset return [unsorted_image, permutation_order-0, class_label]
    test_dataset = JigsawTestDataset(names, labels, patches=patches, img_transformer=img_tr, jig_classes=args.jigsaw_n_classes)
    if args.limit_target and len(test_dataset) > args.limit_target:
        test_dataset = Subset(test_dataset, args.limit_target)
        print("Using %d subset of test dataset" % args.limit_target)
    dataset = ConcatDataset([test_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader


def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop(args.image_size, (args.min_scale, args.max_scale))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))

    tile_tr = []
    if args.tile_random_grayscale:
        tile_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)


def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)

def get_belonged_dataset(domain):
    if domain in digits_datasets:
        return digits_datasets
    elif domain in pacs_datasets:
        return pacs_datasets
    elif domain in vlcs_datasets:
        return vlcs_datasets
    elif domain in office_datasets:
        return office_datasets
    elif domain in office_home_datasets:
        return office_home_datasets
