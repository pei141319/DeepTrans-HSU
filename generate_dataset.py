import numpy as np
import scipy.io as sio
import os

# -------------------------
# Parameters (论文 dc_dataset)
# -------------------------
H, W = 80, 80      # 图像尺寸
L = 200            # 光谱波段数
P = 3              # 端元数
N = H * W

np.random.seed(0)

# -------------------------
# Endmembers (E) - Fe2O3, SiO2, CaO
# -------------------------
wl = np.linspace(1000, 2500, L)
E = np.zeros((L, P))
E[:, 0] = 0.3 + 0.3 * np.exp(-((wl - 1200) ** 2) / 2e5)   # Fe2O3
E[:, 1] = 0.4 + 0.2 * np.exp(-((wl - 1700) ** 2) / 2e5)   # SiO2
E[:, 2] = 0.5 + 0.25 * np.exp(-((wl - 2200) ** 2) / 2e5)  # CaO

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
        y = E @ A[:, i, j]  # 线性混合
        Y[:, i*W + j] = y
Y = Y.T  # shape (H*W, L)

# -------------------------
# 保存数据
# -------------------------
if not os.path.exists('./data'):
    os.makedirs('./data')

sio.savemat('./data/dc_dataset.mat', {
    'Y': Y,
    'A': A.reshape(P, H*W).T,
    'M': E,
    'M1': E
})
print("dc_dataset.mat has been generated correctly in ./data/")
