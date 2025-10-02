# NSARM
NSARM: Next-Scale Autoregressive Modeling for Robust Real-World Image Super-Resolution

<a href='https://arxiv.org/abs/2510.00820'><img src='https://img.shields.io/badge/arXiv-2401.03379-b31b1b.svg'></a> &nbsp;&nbsp;

Authors: [Xiangtao Kong](https://scholar.google.com/citations?hl=en&user=lueNzSgAAAAJ), [Rongyuan Wu](https://scholar.google.com/citations?user=A-U8zE8AAAAJ&hl=en&oi=ao), [Shuaizheng Liu](https://scholar.google.com/citations?hl=en&user=ZCDjTn8AAAAJ), [Lingchen Sun](https://scholar.google.com/citations?hl=en&user=wzdCc-QAAAAJ), and [Lei Zhang](https://scholar.google.com/citations?user=tAK5l1IAAAAJ&hl=en&oi=ao)



## Abstract
Most recent real-world image super-resolution (Real-ISR) methods employ pre-trained text-to-image (T2I) diffusion models to synthesize the high-quality image either from random Gaussian noise, which yields realistic results but is slow due to iterative denoising, or directly from the input low-quality image, which is efficient but at the price of lower output quality. These approaches train ControlNet or LoRA modules while keeping the pre-trained model fixed, which often introduces over-enhanced artifacts and hallucinations, suffering from the robustness to inputs of varying degradations. Recent visual autoregressive (AR) models, such as pre-trained Infinity, can provide strong T2I generation capabilities while offering superior efficiency by using the bitwise next-scale prediction strategy. Building upon next-scale prediction, we introduce a robust Real-ISR framework, namely Next-Scale Autoregressive Modeling (NSARM). Specifically, we train NSARM in two stages: a transformation network is first trained to map the input low-quality image to preliminary scales, followed by an end-to-end full-model fine-tuning. Such a comprehensive fine-tuning enhances the robustness of NSARM in Real-ISR tasks without compromising its generative capability. Extensive quantitative and qualitative evaluations demonstrate that as a pure AR model, NSARM achieves superior visual results over existing Real-ISR methods while maintaining a fast inference speed. Most importantly, it demonstrates much higher robustness to the quality of input images, showing stronger generalization performance.

:star: If NSARM is helpful to your images or projects, please help star this repo. Thanks! :hugs:


## 🔎 Overview framework

![Demo Image](https://github.com/Xiangtaokong/NSARM/blob/main/readme_img/NSARM_Overview.png)

## 📌 Quantitative Results

### NSARM achieves the best general performance in perception metircs across various datasets.

![Demo Image](https://github.com/Xiangtaokong/NSARM/blob/main/readme_img/main_results.png)

### NSARM demonstrates much higher robustness, showing stronger generalization performance.

![Demo Image](https://github.com/Xiangtaokong/NSARM/blob/main/readme_img/robot.png)

### NSARM demonstrates substantial inference speed advantages over compareable methods.


<div align="center">
  <img src="https://github.com/Xiangtaokong/NSARM/blob/main/readme_img/time.png" alt="Demo Image" width="500">
</div>

## 📷 Visual Results

### NSARM achieves superior visual results over existing Real-ISR methods.

![Demo Image](https://github.com/Xiangtaokong/NSARM/blob/main/readme_img/visual_01.png)

<details>
  <summary>Click here to show more visual results.</summary>
  
![Demo Image](https://github.com/Xiangtaokong/NSARM/blob/main/readme_img/morevisual1_01.png)
![Demo Image](https://github.com/Xiangtaokong/NSARM/blob/main/readme_img/morevisual2_01.png)
![Demo Image](https://github.com/Xiangtaokong/NSARM/blob/main/readme_img/morevisual3_01.png)
![Demo Image](https://github.com/Xiangtaokong/NSARM/blob/main/readme_img/morevisual4_01.png)

</details>


## ⚙️ Dependencies and Installation
```
## git clone this repository
git clone https://github.com/Xiangtaokong/NSARM
cd NSARM

# create an environment with python >= 3.10
conda create -n NSARM python=3.10
conda activate NSARM
pip install -r requirements.txt 
```
OR refer to the environment of [BasicSR](https://github.com/XPixelGroup/BasicSR) and [Infinity](https://github.com/FoundationVision/Infinity).

## 🚀 Test

#### Setp 1 Download the pre-trained models

Download NSARM, VAE and T5.

[One Drive]()

[Baidu Drive]().    Key: 

Currently, there are some permission issues with the weight files upload. We will complete the upload as soon as possible.

#### Setp 2 Edit the test script

Edit the file `NSARM/scripts/infer.sh`. 

Please modify the path of your model and data, mainly including:
```
infinity_model_path=
vae_path=
text_encoder_ckpt=
--input_info 
--save_dir 
```
#### Setp 3 Run the command

```
cd NSARM
bash scripts/infer.sh
```
The results will be put in your `--save_dir `.

#### The training code will be released after paper submission.


## ❤️ Acknowledgments
This project is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) and  [Infinity](https://github.com/FoundationVision/Infinity).

## 📧 Contact
If you have any questions, please feel free to contact: `xiangtao.kong@connect.polyu.hk`

## 🎓Citations
If our code helps your research or work, please consider citing our paper.
The following are BibTeX references:

```

```

## 🎫 License
This project is released under the [Apache 2.0 license](LICENSE).
