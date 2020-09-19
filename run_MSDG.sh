source /home/users/hlli/anaconda3/bin/activate jigsaw

pacs=('photo' 'art_painting' 'cartoon' 'sketch')
vlcs=('CALTECH' 'LABELME' 'PASCAL' 'SUN')
officeH=('Art' 'Clipart' 'Product' 'RealWorld')
digits=('mnist', 'mnist_m', 'usps', 'svhn', 'synth')

if [[ ! -z $(printf '%s\n' "${pacs[@]}" | grep -w $1) ]]; then
  domains=('photo' 'art_painting' 'cartoon' 'sketch')
  classes=7
elif [[ ! -z $(printf '%s\n' "${vlcs[@]}" | grep -w $1) ]]; then
  domains=('CALTECH' 'LABELME' 'PASCAL' 'SUN')
  classes=5
elif [[ ! -z $(printf '%s\n' "${officeH[@]}" | grep -w $1) ]]; then
  domains=('Art' 'Clipart' 'Product' 'RealWorld')
  classes=65
else
  domains=('mnist', 'mnist_m', 'usps', 'svhn', 'synth')
  classes=10
fi

targets=($1)
sources=($(comm -3 <(printf "%s\n" "${domains[@]}" | sort) <(printf "%s\n" "${targets[@]}" | sort) | sort -n))
sd=$(printf '%s ' "${sources[@]}")
echo sources-${sd}
echo target-$1


eps=30
nes=False

if [ $2 == "caffenet" ]; then
  imgsize=225
  hflip=0.0
  jitter=0.0
  jw=0.9
  bias=0.7
  nes=True
elif [ $2 == "resnet18" ]; then
  imgsize=222
  hflip=0.5
  jitter=0.4
  jw=0.7
  bias=0.9
else # LetNet for MNIST dataset, params not given by author
  imgsize=32
  hflip=0.0
  jitter=0.0
  eps=32
  jw=0.7 
  bias=0.9
fi


python train_jigsaw.py \
 --epochs ${eps} \
 --n_classes ${classes} \
 --learning_rate 0.001 \
 --nesterov ${nes} \
 --val_size 0.1 \
 --folder_name test \
 --jigsaw_n_classes 30 \
 --min_scale 0.8 --max_scale 1.0 \
 --random_horiz_flip ${hflip} --jitter ${jitter} \
 --tile_random_grayscale 0.1 \
 --source ${sd} --target $1 \
 --network $2 \
 --batch_size $3 \
 --jig_weight ${jw} --bias_whole_image ${bias} \
 --image_size ${imgsize} \
 --gpu $4 \
 --seed 9963