

import sys
sys.path.append("/home/notebook/data/group/kxt/NSARM")
from infinity.dataset.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
import math
import torchvision.utils as vutils
from PIL import Image as PImage
import torch
from torch.nn import functional as F
import gc
import json
import math
import os
import random
import sys
import time
from infinity.dataset.degradations import circular_lowpass_kernel, random_mixed_kernels
import numpy as np
from infinity.utils.diffjpeg import DiffJPEG
from infinity.utils.img_process_util import USMSharp,filter2D
from torchvision.transforms.functional import to_tensor


# blur settings for the first degradation
blur_kernel_size = 21
kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]  # a list for each kernel probability
blur_sigma = [0.2, 3]
betag_range = [0.5, 4] # betag used in generalized Gaussian blur kernels
betap_range = [1, 2]  # betap used in plateau blur kernels
sinc_prob = 0.1  # the probability for sinc filters

# blur settings for the second degradation
blur_kernel_size2 = 21
kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
blur_sigma2 = [0.2, 1.5]
betag_range2 = [0.5, 4]
betap_range2 = [1, 2]
sinc_prob2 = 0.1

# a final sinc filter
final_sinc_prob = 0.8

kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
# TODO: kernel range is now hard-coded, should be in the configure file
pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
pulse_tensor[10, 10] = 1



def gen_kernel():
    # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < sinc_prob:
        # this sinc filter setting is for kernels ranging from [7, 21]
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            kernel_list,
            kernel_prob,
            kernel_size,
            blur_sigma,
            blur_sigma, [-math.pi, math.pi],
            betag_range,
            betap_range,
            noise_range=None)
    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < sinc_prob2:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel2 = random_mixed_kernels(
            kernel_list2,
            kernel_prob2,
            kernel_size,
            blur_sigma2,
            blur_sigma2, [-math.pi, math.pi],
            betag_range2,
            betap_range2,
            noise_range=None)

    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------------------- the final sinc kernel ------------------------------------- #
    if np.random.uniform() < final_sinc_prob:
        kernel_size = random.choice(kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel)
    else:
        sinc_kernel = pulse_tensor
    
    kernel = torch.FloatTensor(kernel)
    kernel2 = torch.FloatTensor(kernel2)
    
    return kernel,kernel2,sinc_kernel

def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    # print(f'im size {im.shape}')
    return im.add(im).add_(-1)


jpeger = DiffJPEG(differentiable=False) # simulate JPEG compression artifacts
usm_sharpener = USMSharp()

img_dir='/home/notebook/data/group/kxt/real_sr_data/testdata_1024/DIV2K_val/GT'
save_dir='/home/notebook/data/group/kxt/real_sr_data/testdata_1024/DIV2K_val/LR_realsergan_3'
ttt='realesrgan'

os.makedirs(save_dir, exist_ok=True)

#realesrgan light

for p in os.listdir(img_dir):
    img_path=os.path.join(img_dir,p)
    save_path=os.path.join(save_dir,p)
    kernel,kernel2,sinc_kernel = gen_kernel()

    with open(img_path, 'rb') as f:
        img: PImage.Image = PImage.open(f)
        img = img.convert('RGB')

        img_B3HW = transform(img, 1024, 1024)
        inp=img_B3HW
        if ttt == 'realesrgan':
            # the first degradation process
            resize_prob = [0.2, 0.7, 0.1]  # up, down, keep
            resize_range = [0.15, 1.5]
            gaussian_noise_prob = 0.5
            noise_range = [1, 30]
            poisson_scale_range = [0.05, 3]
            gray_noise_prob = 0.4
            jpeg_range = [30, 95]

            # the second degradation process
            second_blur_prob = 0.8
            resize_prob2 = [0.3, 0.4, 0.3]  # up, down, keep
            resize_range2 = [0.3, 1.2]
            gaussian_noise_prob2 = 0.5
            noise_range2 = [1, 25]
            poisson_scale_range2 = [0.05, 2.5]
            gray_noise_prob2 = 0.4
            jpeg_range2 = [30, 95]

            final_scale=4

            gt = (inp+1)/2 #从 -1 到 1 变 0-1
            gt=gt.unsqueeze(0)
            # USM sharpen the GT images
            print(gt.shape)

            gt = usm_sharpener(gt)

            kernel1 = kernel
            kernel2 = kernel2
            sinc_kernel = sinc_kernel

            ori_h, ori_w = gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(gt, kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], resize_prob)[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, resize_range[1])
            elif updown_type == 'down':
                scale = np.random.uniform(resize_range[0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = gray_noise_prob
            if np.random.uniform() < gaussian_noise_prob:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=poisson_scale_range,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < second_blur_prob:
                out = filter2D(out, kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], resize_prob2)[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, resize_range2[1])
            elif updown_type == 'down':
                scale = np.random.uniform(resize_range2[0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / final_scale * scale), int(ori_w / final_scale * scale)), mode=mode)
            # add noise
            gray_noise_prob = gray_noise_prob2
            if np.random.uniform() < gaussian_noise_prob2:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=poisson_scale_range2,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // final_scale, ori_w // final_scale), mode=mode)
                out = filter2D(out, sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
                out = torch.clamp(out, 0, 1)
                out = jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
                out = torch.clamp(out, 0, 1)
                out = jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // final_scale, ori_w // final_scale), mode=mode)
                out = filter2D(out, sinc_kernel)

            # clamp and round
            lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        elif ttt=='light':
                    # the first degradation process
            resize_prob = [0.2, 0.7, 0.1]  # up, down, keep
            resize_range = [0.5, 1.5]
            gaussian_noise_prob = 0.3
            noise_range = [1, 20]
            poisson_scale_range = [0.05, 2]
            gray_noise_prob = 0.2
            jpeg_range = [60, 95]

            # the second degradation process
            second_blur_prob = 0.8
            resize_prob2 = [0.3, 0.4, 0.3]  # up, down, keep
            resize_range2 = [0.3, 1.2]
            gaussian_noise_prob2 = 0.5
            noise_range2 = [1, 25]
            poisson_scale_range2 = [0.05, 2.5]
            gray_noise_prob2 = 0.4
            jpeg_range2 = [60, 95]

            final_scale=4

            gt = (inp+1)/2 #从 -1 到 1 变 0-1
            gt=gt.unsqueeze(0)
            # USM sharpen the GT images
            gt = usm_sharpener(gt)

            kernel1 = kernel


            ori_h, ori_w = gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(gt, kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], resize_prob)[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, resize_range[1])
            elif updown_type == 'down':
                scale = np.random.uniform(resize_range[0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = gray_noise_prob
            if np.random.uniform() < gaussian_noise_prob:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=poisson_scale_range,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # # JPEG compression
            # jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
            # out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            # out = jpeger(out, quality=jpeg_p)

            # # ----------------------- The second degradation process ----------------------- #
            # # blur
            # if np.random.uniform() < second_blur_prob:
            #     out = filter2D(out, kernel2)
            # # random resize
            # updown_type = random.choices(['up', 'down', 'keep'], resize_prob2)[0]
            # if updown_type == 'up':
            #     scale = np.random.uniform(1, resize_range2[1])
            # elif updown_type == 'down':
            #     scale = np.random.uniform(resize_range2[0], 1)
            # else:
            #     scale = 1
            # mode = random.choice(['area', 'bilinear', 'bicubic'])
            # out = F.interpolate(
            #     out, size=(int(ori_h / final_scale * scale), int(ori_w / final_scale * scale)), mode=mode)
            # # add noise
            # gray_noise_prob = gray_noise_prob2
            # if np.random.uniform() < gaussian_noise_prob2:
            #     out = random_add_gaussian_noise_pt(
            #         out, sigma_range=noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
            # else:
            #     out = random_add_poisson_noise_pt(
            #         out,
            #         scale_range=poisson_scale_range2,
            #         gray_prob=gray_noise_prob,
            #         clip=True,
            #         rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // final_scale, ori_w // final_scale), mode=mode)
                out = filter2D(out, sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
                out = torch.clamp(out, 0, 1)
                out = jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
                out = torch.clamp(out, 0, 1)
                out = jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // final_scale, ori_w // final_scale), mode=mode)
                out = filter2D(out, sinc_kernel)

            # clamp and round
            lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
            
        # 保存图像
        vutils.save_image(lq, save_path)
        print(f"图像已保存到: {save_path}")
