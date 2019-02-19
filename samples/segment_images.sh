#!/bin/bash

echo ""
echo "Usage: ./segmented_cloud.sh image_addresses.txt"
echo "--"
echo ""

source ~/softwares/tensorflow3/venv3/bin/activate

filename=$1
segmented_image=''
masked_side='none'

n=1
while read -r line; do

	rem=$(( $n % 2 ))
	if [ $rem -eq 0 ]
	then
		echo "$n is even"
	  	depth_image=$line
	  	echo $depth_image
	  	./generate_point_cloud $segmented_image $depth_image $masked_side

	else
	 	echo "$n is odd"
	 	rgb_image=$line
	  	echo $rgb_image

	  	python demo.py $rgb_image > segmented_image.txt
		segmented_image=`cat segmented_image.txt`
	fi
	
	n=$((n+1))
done < $filename

rm segmented_image.txt