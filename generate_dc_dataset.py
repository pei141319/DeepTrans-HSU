import numpy as np
import scipy.io as sio
import os

# -------------------------
# Parameters
# -------------------------
H, W = 80, 80       # 图像尺寸
L = 191             # 光谱波段数
P = 6               # 端元数
N = H * W

np.random.seed(0)

# -------------------------
# Endmembers (随机生成6个端元)
# -------------------------
E = np.random.rand(L, P)

# -------------------------
# Abundance maps (A)
# -------------------------
A = np.zeros((P, H, W))
for i in range(4):
    for j in range(4):
        a = np.random.rand(P)
        a = a / a.sum()
        A[:, i*20:(i+1)*20, j*20:(j+1)*20] = a[:, None, None]

# -------------------------
# 构建高光谱数据 Y = E * A
# -------------------------
Y = np.zeros((L, H*W))
for i in range(H):
    for j in range(W):
        Y[:, i*W + j] = E @ A[:, i, j]
Y = Y.T  # shape (H*W, L)

# -------------------------
# 保存为 dc_dataset.mat
# -------------------------
os.makedirs('./data', exist_ok=True)
sio.savemat('./data/dc_dataset.mat', {'Y': Y, 'A': A.reshape(P, H*W).T, 'M': E, 'M1': E})
print("dc_dataset.mat has been generated correctly in ./data/")
