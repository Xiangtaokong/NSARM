#!/bin/bash

# set arguments for inference
# 0.25M
pn=1M
model_type=infinity_8b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
infinity_model_path=/home/notebook/data/group/kxt/Infinity_replace/checkpoints/INF_8B_realesrgan_n4_bs2_g8_cum4_type5_from_200k_INF_NOFlip/ar-ckpt-giter076K-ep5-iter3520-last.pth
vae_type=14
vae_path=/home/notebook/data/group/kxt/Infinity/weights/infinity_vae_d56_f8_14_patchify.pth
cfg=1
tau=0.5
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=/home/notebook/data/group/kxt/Infinity/weights/T5
text_channels=2048
apply_spatial_patchify=1

# run inference
CUDA_VISIBLE_DEVICES=0 python3 tools/run_infinity.py \
--cfg ${cfg} \
--tau ${tau} \
--pn ${pn} \
--model_path ${infinity_model_path} \
--vae_type ${vae_type} \
--vae_path ${vae_path} \
--add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
--use_bit_label ${use_bit_label} \
--model_type ${model_type} \
--rope2d_each_sa_layer ${rope2d_each_sa_layer} \
--rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
--use_scale_schedule_embedding ${use_scale_schedule_embedding} \
--checkpoint_type ${checkpoint_type} \
--text_encoder_ckpt ${text_encoder_ckpt} \
--text_channels ${text_channels} \
--apply_spatial_patchify ${apply_spatial_patchify} \
--prompt "" \
--seed 1 \
--apply_spatial_patchify 1 \
--input_info /home/notebook/data/group/kxt/NSARM/data/test/path_caption \
--save_dir "/home/notebook/data/group/kxt/NSARM/result_img/NSARM" \
--enable_model_cache 1 


# A high-quality image with harmonious colors and rich details.

# /home/notebook/data/group/kxt/Infinity/weights/infinity_8b_weights

#/home/notebook/data/group/kxt/Infinity/weights/INF_8B_fromtest.pth

#tau=0.5 默认