source /home/users/hlli/anaconda3/bin/activate jigsaw

domains=('photo' 'art_painting' 'cartoon' 'sketch')
for sd in ${domains[@]}; do
	for td in ${domains[@]}; do
		if [ $sd = $td ]
		then
			echo "Skip: $sd -> $td"
		else
			echo "$sd -> $td"
                        python train_jigsaw.py --batch_size 128 --n_classes 7 --learning_rate 0.001 --network caffenet --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all --TTA --image_size 225 --nesterov --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0 --jitter 0 --tile_random_grayscale 0.1 --source $sd --target $td --jig_weight 0.0 --bias_whole_image 1.0 2>/dev/null | awk 'END {print}'
			echo $res
		fi
	done
done
echo "End"
