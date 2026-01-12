# -*- coding: utf-8 -*-
import numpy as np
import os
import scipy.io as sio
from scipy.optimize import nnls
import datasets
import utils
import plots  # <--- [核心] 导入绘图模块，用于生成 PNG 图片

# =========================================================================
# 1. 内置算法实现 (VCA & FCLS) - 无需安装第三方库
# =========================================================================

def vca_algorithm(Y, R):
    """
    VCA算法手动实现 (Vertex Component Analysis)
    :param Y: 数据矩阵 (L波段 x N像素)
    :param R: 端元数量
    :return: 提取的端元矩阵 (L x R), 端元索引
    """
    L, N = Y.shape
    
    # --- SVD 降维 ---
    # 计算相关矩阵 Y*Y^T 的 SVD
    correlation_matrix = np.dot(Y, Y.T) / N
    U, S, V = np.linalg.svd(correlation_matrix)
    
    # 取前 R 个主成分构建投影矩阵
    Ud = U[:, :R] # (L, R)
    
    # 将数据投影到 R 维子空间
    x_p = np.dot(Ud.T, Y) # (R, N)
    
    # --- VCA 迭代寻找端元 ---
    indices = np.zeros(R, dtype=int)
    
    # 初始化辅助矩阵 A (用于存储已找到端元的投影)
    # 关键修复: 必须初始化为 (R, R) 零矩阵
    A = np.zeros((R, R))
    
    for i in range(R):
        # 生成随机向量 w
        w = np.random.rand(R, 1)
        f = w
        
        # 正交投影：将 w 投影到已找到端元的正交补空间
        if i > 0:
            A_curr = A[:, :i] # 取前 i 列
            # 计算投影系数: coeff = (A^T A)^-1 A^T w
            # 使用 lstsq 求解更稳健
            coeff = np.linalg.lstsq(A_curr, w, rcond=None)[0]
            # 减去投影分量
            f = w - np.dot(A_curr, coeff)
        
        # 归一化
        length = np.linalg.norm(f)
        if length > 1e-9:
            f = f / length
        
        # 投影数据并寻找极值点
        v = np.dot(f.T, x_p)
        ind = np.argmax(np.abs(v))
        indices[i] = ind
        
        # 将找到的点加入 A 矩阵
        A[:, i] = x_p[:, ind]
        
    # 根据索引提取原始端元
    Ae = Y[:, indices]
    return Ae, indices

def fcls_algorithm(M, Y):
    """
    FCLS算法手动实现 (Fully Constrained Least Squares)
    :param M: 端元矩阵 (L x P)
    :param Y: 图像数据 (L x N)
    :return: 丰度矩阵 (N x P)
    """
    L, P = M.shape
    _, N = Y.shape
    
    # 增广矩阵法: 强制和为一 (ASC)
    # 添加一行权重为 delta 的 1
    delta = 20
    M_aug = np.vstack((delta * M, np.ones((1, P))))
    Y_aug = np.vstack((delta * Y, np.ones((1, N))))
    
    A_est = np.zeros((N, P))
    
    print(f"   [FCLS] 正在逐像素计算丰度 ({N} 像素)...")
    
    # 逐像素求解非负最小二乘 (NNLS)
    for i in range(N):
        # nnls 自动满足非负性 (ANC)
        coef, _ = nnls(M_aug, Y_aug[:, i])
        A_est[i, :] = coef
        
    return A_est

# =========================================================================
# 2. 对比实验管理类
# =========================================================================

class TraditionalBaselines:
    def __init__(self, dataset_name, device, save_root="comparison_results"):
        """
        初始化：加载数据并创建保存目录
        """
        # 加载数据
        self.data_obj = datasets.Data(dataset=dataset_name, device=device)
        self.dataset_name = dataset_name
        
        # 获取数据并转为 Numpy 格式
        # Y 需要转置为 (Bands, Pixels) 适配 VCA/FCLS
        self.Y_tensor = self.data_obj.get("hs_img") 
        self.Y = self.Y_tensor.cpu().numpy().T 
        
        self.M_true = self.data_obj.get("end_mem").numpy()
        self.A_true = self.data_obj.get("abd_map").cpu().numpy()
        
        self.P = self.data_obj.P
        self.col = self.data_obj.col
        self.L = self.data_obj.L

        # 创建保存路径 (注意末尾加斜杠，适配 plots.py 的路径拼接)
        self.save_dir = os.path.join(save_root, dataset_name, "VCA_FCLS/")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def run_vca_fcls(self):
        print(f"\n[Baseline] 启动 VCA+FCLS 实验: {self.dataset_name}")
        
        # --- 1. 执行算法 ---
        # VCA 提取端元
        E_est, _ = vca_algorithm(self.Y, self.P)
        print("   -> 端元提取完成 (VCA)")
        
        # FCLS 估计丰度
        A_est_flat = fcls_algorithm(E_est, self.Y)
        print("   -> 丰度估计完成 (FCLS)")
        
        # --- 2. 数据整形与对齐 ---
        # 将丰度 reshape 回 (H, W, P)
        A_est_map = A_est_flat.reshape(self.col, self.col, self.P)
        A_true_map = self.A_true.reshape(self.col, self.col, self.P)
        
        # --- 3. 计算评价指标 ---
        # 计算 SAD (光谱角距离)
        _, mSAD = utils.compute_sad(E_est, self.M_true)
        # 计算 RMSE (均方根误差)
        _, mRMSE = utils.compute_rmse(A_true_map, A_est_map)
        
        # --- 4. 保存数值结果 (.mat) ---
        mat_path = self.save_dir + f"{self.dataset_name}_vca_fcls.mat"
        sio.savemat(mat_path, {
            "E_est": E_est,
            "A_est": A_est_map,
            "mSAD": mSAD,
            "mRMSE": mRMSE
        })
        print(f"   -> 数据已保存: {mat_path}")
        
        # --- 5. 生成可视化图片 (PNG) ---
        print("   -> 正在生成对比图 (Abundance & Endmembers)...")
        
        # (A) 绘制丰度图对比
        plots.plot_abundance(
            ground_truth=A_true_map, 
            estimated=A_est_map, 
            em=self.P, 
            save_dir=self.save_dir, 
            dataset=self.dataset_name
        )
        
        # (B) 绘制端元波形对比
        plots.plot_endmembers(
            target=self.M_true, 
            pred=E_est, 
            em=self.P, 
            save_dir=self.save_dir, 
            dataset=self.dataset_name
        )
        
        # 保存简单的文本报告
        with open(self.save_dir + "metrics.txt", "w") as f:
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Method: VCA + FCLS\n")
            f.write(f"Mean SAD: {mSAD:.4f}\n")
            f.write(f"Mean RMSE: {mRMSE:.4f}\n")

        print(f"[Baseline] 全部完成! 可视化结果请查看: {self.save_dir}")
        return {"mSAD": mSAD, "mRMSE": mRMSE}