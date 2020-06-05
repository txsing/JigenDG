source /home/users/hlli/anaconda3/bin/activate jigsaw

domains=('photo' 'art_painting' 'cartoon' 'sketch')
for sd in ${domains[@]}; do
	for td in ${domains[@]}; do
		if [ $sd = $td ]
		then
			echo "Skip: $sd -> $td"
		else
			echo "$sd -> $td"
			python train_jigsaw.py --epochs 30 --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --TTA --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --source $sd --target $td --jig_weight 0.7 --bias_whole_image 0.9 --image_size 222 2>/dev/null | awk 'END {print}' 
			echo $res
		fi
	done
done
echo "End"
