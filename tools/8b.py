import random
import torch
torch.cuda.set_device(2)
import cv2
import numpy as np
from tools.run_infinity import *

model_path='weights/infinity_8b_weights'
vae_path='weights/infinity_vae_d56_f8_14_patchify.pth'
text_encoder_ckpt = 'weights/flan-t5-xl-official'
args=argparse.Namespace(
    pn='1M',
    model_path=model_path,
    cfg_insertion_layer=0,
    vae_type=14,
    vae_path=vae_path,
    add_lvl_embeding_only_first_block=1,
    use_bit_label=1,
    model_type='infinity_8b',
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0,
    sampling_per_bits=1,
    text_encoder_ckpt=text_encoder_ckpt,
    text_channels=2048,
    apply_spatial_patchify=1,
    h_div_w_template=1.000,
    use_flex_attn=0,
    cache_dir='/dev/shm',
    checkpoint_type='torch_shard',
    seed=0,
    bf16=1,
    save_file='tmp.jpg'
)