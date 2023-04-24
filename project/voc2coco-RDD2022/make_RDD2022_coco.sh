#!/bin/bash

RDD2022_PATH=$1

if [ ! -d "$RDD2022_PATH" ]; then
    echo "Error: $RDD2022_PATH is not a valid path"
    exit 1
fi

INPUT_ANNOTATIONS_PATH=${RDD2022_PATH}/annotations

OUTPUT_DIR_ANNOTATIONS=RDD2022_COCO/annotations

if [ ! -d $OUTPUT_DIR_ANNOTATIONS ]; then
    mkdir -p $OUTPUT_DIR_ANNOTATIONS
fi


find $INPUT_ANNOTATIONS_PATH -type f | shuf > temp_path_list.txt

IMAGE_COUNT=$(wc -l < temp_path_list.txt)
VAL_IMAGE_COUNT=$(echo "scale=0; $IMAGE_COUNT * 0.25 / 1" | bc)
#VAL_IMAGE_COUNT=0
TRAIN_IMAGE_COUNT=$(echo "$IMAGE_COUNT - $VAL_IMAGE_COUNT" | bc)

echo "Total images: $IMAGE_COUNT"
echo "Train images: $TRAIN_IMAGE_COUNT"
echo "Val images: $VAL_IMAGE_COUNT"

head -n $TRAIN_IMAGE_COUNT temp_path_list.txt > temp_path_list_train.txt
tail -n $VAL_IMAGE_COUNT temp_path_list.txt > temp_path_list_val.txt

python voc2coco.py \
    --ann_paths_list temp_path_list_train.txt \
    --labels labels.txt \
    --output $OUTPUT_DIR_ANNOTATIONS/train.json \
    --extract_num_from_imgid

python voc2coco.py \
    --ann_paths_list temp_path_list_val.txt \
    --labels labels.txt \
    --output $OUTPUT_DIR_ANNOTATIONS/val.json \
    --extract_num_from_imgid

rm temp_path_list.txt
rm temp_path_list_train.txt
rm temp_path_list_val.txt

echo "Done!"
