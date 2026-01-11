from datasets import load_dataset
import scipy.io as sio
import os
import numpy as np

# 1. 加载数据集
print("正在从 Hugging Face 下载 Jasper Ridge 数据集...")
ds = load_dataset("danaroth/jasper_ridge")

# 2. 提取数据内容 (根据该数据集在 HF 上的结构)
# 注意：该数据集通常包含 'train' 拆分
data_dict = ds['train'][0] 

# 3. 构建符合你项目代码要求的字典结构
# 你的代码需要：Y (数据), A (丰度), M (端元), M1 (初始权重)
output_data = {
    'Y': np.array(data_dict['Y']), 
    'A': np.array(data_dict['A']),
    'M': np.array(data_dict['M']),
    'M1': np.array(data_dict['M1']) if 'M1' in data_dict else np.array(data_dict['M'])
}

# 4. 创建文件夹并保存为 .mat 文件
save_path = "./data"
if not os.path.exists(save_path):
    os.makedirs(save_path)

sio.savemat(os.path.join(save_path, "jasper_dataset.mat"), output_data)
print(f"成功！数据集已保存至: {save_path}/jasper_dataset.mat")