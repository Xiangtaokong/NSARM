import os
import glob
import json
import torch
from PIL import Image
from tqdm import tqdm
from llava.llava_agent import LLavaAgent


from llava.llava_agent import LLavaAgent
from CKPT_PTH import LLAVA_MODEL_PATH
import torch


# 设置设备
if torch.cuda.device_count() >= 2:
    device = 'cuda:1'
else:
    device = 'cuda:0'
    
# 加载LLaVA模型
llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=device, load_8bit=False, load_4bit=False)

def annotate_images_with_llava(input_folder, output_file="image_annotations.txt"):
    """
    使用LLaVA模型对图像进行标注
    """


    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 打开输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        # 处理每个图像
        for img_path in tqdm(sorted(image_files), desc="标注图像"):
            try:
                # 获取文件名
                filename = os.path.basename(img_path)
                
                # 加载图像
                image = Image.open(img_path).convert("RGB")
                # 调整大小
                #image = image.resize((256, 256), Image.BICUBIC)
                
                # 使用LLaVA生成描述
                captions = llava_agent.gen_image_caption([image])
                des = captions[0] if captions else ""
                
                # 创建JSON对象
                json_obj = {
                    "image_path": img_path,
                    "h_div_w": 1,
                    "long_caption": des,
                    "long_caption_type": "llava-v1.5-13b"
                }
                
                # 写入JSONL文件
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                
                # # 保存处理后的图像
                # save_path = os.path.join('/home/notebook/data/group/kxt/real_sr_data/testdata_1024/DrealSR', filename)
                # image.save(save_path)
                
                # print(f"已处理: {img_path}")
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
                continue
    
    print(f"标注完成! 结果已保存到 {output_file}")
    return True

if __name__ == "__main__":
    # # 设置输入文件夹
    # input_folder = '/home/notebook/data/group/kxt/real_sr_data/testdata_1024/DIV2K_val/Bicx4'
    # output_file = "/home/notebook/data/group/kxt/real_sr_data/meta_results/DIV2K_Bicx4.jsonl"
    
    # # 执行标注
    # annotate_images_with_llava(input_folder, output_file)

    dir_list=[
       
        # ['/home/notebook/data/group/kxt/real_sr_data/testdata_1024/DIV2K_val/Bicx4_Niose40','/home/notebook/data/group/kxt/real_sr_data/meta_results/DIV2K_Bicx4_Niose40.jsonl'],
        # ['/home/notebook/data/group/kxt/real_sr_data/testdata_1024/DIV2K_val/Blur2_Bicx4','/home/notebook/data/group/kxt/real_sr_data/meta_results/DIV2K_Blur2_Bicx4.jsonl'],
        # ['/home/notebook/data/group/kxt/real_sr_data/testdata_1024/DIV2K_val/Blur2_Bicx4_Niose40_Jpeg50','/home/notebook/data/group/kxt/real_sr_data/meta_results/DIV2K_Blur2_Bicx4_Niose40_Jpeg50.jsonl'],
        # ['/home/notebook/data/group/kxt/real_sr_data/testdata_1024/DIV2K_val/LR_realsergan','/home/notebook/data/group/kxt/real_sr_data/meta_results/DIV2K_realsergan.jsonl'],
        # ['/home/notebook/data/group/kxt/real_sr_data/testdata_1024/DRealSR/test_LR_256','/home/notebook/data/group/kxt/real_sr_data/meta_results/DRealSR_256.jsonl'],
        # ['/home/notebook/data/group/kxt/real_sr_data/testdata_1024/RealSR/test_LR_256','/home/notebook/data/group/kxt/real_sr_data/meta_results/RealSR_256.jsonl'],
        # ['/home/notebook/data/group/kxt/real_sr_data/testdata_1024/RP60_256','/home/notebook/data/group/kxt/real_sr_data/meta_results/RP60_256.jsonl'],
        # ['/home/notebook/data/group/kxt/real_sr_data/testdata_1024/DRealSR/test_LR_128','/home/notebook/data/group/kxt/real_sr_data/meta_to512/DRealSR_128.jsonl'],
        # ['/home/notebook/data/group/kxt/real_sr_data/testdata_1024/RealSR/test_LR_128','/home/notebook/data/group/kxt/real_sr_data/meta_to512/RealSR_128.jsonl'],
        # ['/home/notebook/data/group/kxt/real_sr_data/testdata_1024/RP60_256','/home/notebook/data/group/kxt/real_sr_data/meta_to512/RP60_256.jsonl'],
        ['/home/notebook/data/group/kxt/dataset/DF2K/LR256','/home/notebook/data/group/kxt/dataset/DF2K/meta_info/LR_128.jsonl'],
        # ['/home/notebook/data/group/kxt/real_sr_data/testdata_1024/RealSR/test_LR_128','/home/notebook/data/group/kxt/real_sr_data/meta_to512/RealSR_128.jsonl'], 
    ]
    for i in dir_list:
        annotate_images_with_llava(i[0],i[1])
