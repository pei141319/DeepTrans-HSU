import sys
import os

# 确保加载官方 datasets 库，避免与本地 datasets.py 冲突
current_dir = os.getcwd()
if current_dir in sys.path:
    sys.path.remove(current_dir)

from datasets import load_dataset
import scipy.io as sio
import numpy as np

print("正在从镜像站下载 Jasper Ridge 数据集...")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 加载数据集
ds = load_dataset("danaroth/jasper_ridge")
# 获取数据字典 (通常在 'train' 拆分中)
raw_data = ds['train'][0]

# 自动映射键值对
# 你的 datasets.py 期望的键是 Y, A, M, M1
output_data = {}

# 1. 寻找光谱数据 Y
for k in ['Y', 'data', 'img', 'hsi']:
    if k in raw_data:
        output_data['Y'] = np.array(raw_data[k])
        break

# 2. 寻找丰度图 A
for k in ['A', 'ground_truth', 'abundance', 'abd']:
    if k in raw_data:
        output_data['A'] = np.array(raw_data[k])
        break

# 3. 寻找端元 M
for k in ['M', 'endmembers', 'end']:
    if k in raw_data:
        output_data['M'] = np.array(raw_data[k])
        break

# 4. 设置初始权重 M1 (若无则用 M 代替)
output_data['M1'] = np.array(raw_data.get('M1', output_data.get('M')))

# 检查是否找全了必要变量
missing = [k for k in ['Y', 'A', 'M'] if k not in output_data]
if missing:
    print(f"错误：无法在数据集中找到变量: {missing}")
    print("当前数据集包含的键有:", raw_data.keys())
else:
    save_path = "./data"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    file_name = os.path.join(save_path, "jasper_dataset.mat")
    sio.savemat(file_name, output_data)
    print(f"成功！文件已保存至: {file_name}")
    print(f"数据维度确认 - Y: {output_data['Y'].shape}, A: {output_data['A'].shape}")

