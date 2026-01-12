# import random
# import torch
# import numpy as np
# import Trans_mod
#
# seed = 1
# random.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)
#
# # Device Configuration
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print("\nSelected device:", device, end="\n\n")
#
# tmod = Trans_mod.Train_test(dataset='dc', device=device, skip_train=False, save=True)
# tmod.run(smry=False)
# -*- coding: utf-8 -*-
# """
# DeepTrans-HSU项目启动脚本
# 功能：初始化实验环境，启动高光谱解混模型的训练+测试流程
# 适用场景：基于CNN-Transformer的高光谱遥感图像解混（Samson/Jasper Ridge数据集）
# """
# # 导入Python原生随机模块 - 用于控制随机数生成，保证实验可复现
# import random
# # 导入PyTorch深度学习框架 - 核心：模型搭建、张量计算、GPU加速
# import torch
# # 导入NumPy数值计算库 - 核心：高光谱三维数据立方体的处理（读取/预处理/数值运算）
# import numpy as np
# # 导入项目自定义核心模块 - 封装了Train_test类，包含模型搭建、训练、测试、指标计算（SAD/RMSE）等逻辑
# import Trans_mod

# # -------------------------
# # 固定随机种子（实验可复现性关键）
# # 高光谱解混实验中，随机种子固定能避免因参数初始化/数据打乱导致结果不一致
# # -------------------------

# seed = 1  # 随机种子值（可自定义，如42、100，保持固定即可）
# random.seed(seed)  # 固定Python原生随机数种子
# torch.manual_seed(seed)  # 固定PyTorch CPU/GPU随机数种子（模型参数初始化/Dropout等）
# np.random.seed(seed)  # 固定NumPy随机数种子（数据预处理/数据增强等）

# # -------------------------
# # 指定计算设备（GPU优先，加速训练）
# # 高光谱数据维度高（数百个波段），Transformer模块计算量大，GPU能大幅缩短训练时间
# # -------------------------
# # 检测是否有可用的NVIDIA GPU，有则用cuda:0（第一块GPU），无则用CPU
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# # 打印选中的设备，方便确认是否成功调用GPU（关键：无GPU时训练会极慢）
# print("Selected device:", device)

# # -------------------------
# # 初始化训练测试类 + 执行核心流程
# # Train_test类是项目核心，封装了「数据加载→模型搭建→训练→测试→结果保存」全流程
# # -------------------------
# # 实例化Train_test类，配置实验参数
# tmod = Trans_mod.Train_test(
#     dataset='samson',  # 指定训练数据集：'dc'为测试用数据集，毕设需改为'samson'/'jasper/apex'（需匹配Trans_mod内的数据集映射）
#     device=device,  # 指定模型训练/测试的计算设备（上述选中的cuda:0/cpu）
#     skip_train=False,  # 是否跳过训练：False=从头训练；True=加载已保存模型直接测试（训练完成后可改True）
#     save=True  # 是否保存结果：True=保存模型权重、SAD/RMSE指标、可视化图；False=仅临时运行不保存
# )

# # 执行训练+测试核心流程
# # smry=False：不打印模型结构摘要；改为True可输出CNN-Transformer各层的参数/维度（调试模型架构时建议开启）
# tmod.run(smry=False)
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import random
import torch
import numpy as np
import Trans_mod
import os
import csv
import datetime
from vca_fcls_baseline import TraditionalBaselines 

# ==========================================
# 核心功能：写入全局实验记录表
# ==========================================
def log_to_csv(dataset, model_name, sad, rmse, filename="all_experiments_results.csv"):
    """
    将实验结果追加到根目录的 CSV 文件中，方便 Excel 打开对比
    """
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 1. 如果是新文件，先写入表头 (像 Excel 的第一行)
        if not file_exists:
            writer.writerow(['Time', 'Dataset', 'Model', 'SAD (Spectral)', 'RMSE (Abundance)', 'Status'])
        
        # 2. 获取当前时间
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 3. 写入数据行
        writer.writerow([now, dataset, model_name, f"{sad:.4f}", f"{rmse:.4f}", "Success"])
        
    print(f"✅ 结果已归档到: {filename}")

# -------------------------
# 基础配置
# -------------------------
seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Selected device:", device)

# >>> 在这里修改数据集，之前的记录会自动保留 <<<
dataset_name = 'samson' 

# -------------------------
# 1. 运行 Baseline (VCA+FCLS)
# -------------------------
print(f"\n>>> [1/2] Running VCA+FCLS on {dataset_name}...")
baseline_tool = TraditionalBaselines(dataset_name=dataset_name, device=device)
vca_res = baseline_tool.run_vca_fcls()

# 记录 VCA 结果
log_to_csv(dataset_name, "VCA+FCLS", vca_res['mSAD'], vca_res['mRMSE'])

# -------------------------
# 2. 运行 Transformer 模型
# -------------------------
print(f"\n>>> [2/2] Running DeepTrans-HSU on {dataset_name}...")
tmod = Trans_mod.Train_test(
    dataset=dataset_name, 
    device=device, 
    skip_train=False, 
    save=True
)

# 运行并获取返回值 (需要您已完成第一步修改)
try:
    trans_sad, trans_rmse = tmod.run(smry=False)
    # 记录 Transformer 结果
    log_to_csv(dataset_name, "CNN-Transformer", trans_sad, trans_rmse)
    
except TypeError:
    print("\n❌ 错误：Trans_mod.py 没有返回指标。")
    print("请确保您在 Trans_mod.py 的 run 函数最后加了 'return mean_sad, mean_rmse'")

print("\n" + "="*50)
print(f"实验结束！请查看根目录下的 all_experiments_results.csv")
print("="*50)