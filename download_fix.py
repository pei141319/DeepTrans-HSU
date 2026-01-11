import sys
import os

# 强制移除当前目录路径，确保加载官方 datasets 库
current_dir = os.getcwd()
if current_dir in sys.path:
    sys.path.remove(current_dir)

from datasets import load_dataset
import scipy.io as sio
import numpy as np

# 重新添加当前目录以便后续操作
sys.path.append(current_dir)

print("正在下载 Jasper Ridge 数据集...")
# 设置镜像站以防网络问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

ds = load_dataset("danaroth/jasper_ridge")
data_dict = ds['train'][0]

output_data = {
    'Y': np.array(data_dict['Y']), 
    'A': np.array(data_dict['A']),
    'M': np.array(data_dict['M']),
    'M1': np.array(data_dict['M1']) if 'M1' in data_dict else np.array(data_dict['M'])
}

save_path = "./data"
if not os.path.exists(save_path):
    os.makedirs(save_path)

file_name = os.path.join(save_path, "jasper_dataset.mat")
sio.savemat(file_name, output_data)
print(f"成功！文件已保存至: {file_name}")
