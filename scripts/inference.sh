#!/bin/bash

dataset=ego4d  # ego4d or epic_kitchens
checkpoint=/coc/flash5/myang415/objects_project_clean/objects_project_localization/392829/objects_model_latest.pth  # Path to finetuned checkpoint if evaluation. If training, use pretrained model checkpoint.

data_dir=/coc/flash5/datasets/ego4d/ego4dsounds_540p_correct_shape
eval_data_dir=/coc/flash5/datasets/ego4d/ego4dsounds_540p_correct_shape

mask_dir=/coc/flash5/datasets/ego4d/ego4dsounds_masks_npz
eval_mask_dir=/coc/flash5/datasets/ego4d/ego4d_eval_object_detection_masks_npz
eval_pool_mask_dir=/coc/flash5/datasets/ego4d/ego4d_object_detection_pool_masks

train_metadata_file=/coc/flash5/myang415/objects_project_clean/metadata_files/ego4dsounds_train.csv
eval_metadata_file=/coc/flash5/myang415/objects_project_clean/metadata_files/ego4d_object_detection_eval_1.2k.csv


torchrun train_detection.py \
    --normalize_inner_prod \
    --ast_imagenet_pretrain \
    --freeze_slots \
    --dataset $dataset \
    --epochs 16 \
    --checkpoint $checkpoint \
    --data_dir $data_dir \
    --eval_data_dir $eval_data_dir \
    --mask_dir $mask_dir \
    --eval_mask_dir $eval_mask_dir \
    --eval_pool_mask_dir $eval_pool_mask_dir \
    --train_metadata_file $train_metadata_file \
    --eval_metadata_file $eval_metadata_file \
    --eval  # Remove this flag to train the model. Keep to only run evaluation.
