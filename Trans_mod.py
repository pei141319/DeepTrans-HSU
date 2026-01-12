# -*- coding: utf-8 -*-
"""
Trans_mod.py - 项目核心训练测试模块（高光谱解混专属）
核心功能：
1. 定义AutoEncoder模型（CNN编码器 + ViT特征提取 + CNN解码器）
2. 定义NonZeroClipper类，实现模型参数非负约束
3. 定义Train_test类，封装完整的模型训练、测试、结果保存与可视化流程
依赖模块：datasets（数据加载）、transformer（ViT实现）、utils（指标计算）、plots（结果可视化）
"""
# 导入系统模块：目录创建、数据序列化、计时
import os
import pickle
import time

# 导入数据处理模块：读取.mat文件、PyTorch核心、神经网络层
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
# from torchsummary import summary  # 模型结构摘要（注释未启用，需额外安装）

# 导入项目自定义模块
import datasets  # 数据集加载模块（对应datasets.py）
import plots     # 结果可视化模块（对应plots.py）
import transformer  # ViT网络实现（对应transformer.py）
import utils     # 工具函数（指标计算、损失函数，对应utils.py）

# 光谱平滑性损失类
class SpectralSmoothnessLoss(torch.nn.Module):
    def __init__(self):
        super(SpectralSmoothnessLoss, self).__init__()

    def forward(self, endmembers):
        """
        计算端元光谱的平滑性损失
        endmembers: (L, P) - L是波段数，P是端元数
        """
        # 计算相邻波段间的差分（二阶差分，更严格的平滑性）
        if endmembers.shape[0] < 3:
            # 如果波段数少于3，则只计算一阶差分
            if endmembers.shape[0] < 2:
                return torch.tensor(0.0, device=endmembers.device, requires_grad=True)
            diff1 = endmembers[1:, :] - endmembers[:-1, :]
            smoothness_loss = torch.mean(diff1 ** 2)
        else:
            # 计算一阶差分
            diff1 = endmembers[1:, :] - endmembers[:-1, :]
            # 计算二阶差分
            diff2 = diff1[1:, :] - diff1[:-1, :]
            # 使用L2范数计算平滑性损失
            smoothness_loss = torch.mean(diff2 ** 2)
        return smoothness_loss

# -------------------------
# AutoEncoder类：CNN-ViT混合自编码器（高光谱解混核心模型）
# 作用：实现高光谱图像的编码、特征提取、丰度预测与图像重构
# -------------------------
class AutoEncoder(nn.Module):
    def __init__(self, P, L, size, patch, dim):
        """
        初始化自编码器模型
        :param P: int - 端元数（地物类别数，如Samson=3、DC=6）
        :param L: int - 高光谱波段数（如Samson=156、DC=191）
        :param size: int - 高光谱图像的空间尺寸（如Samson=95 → 95×95像素）
        :param patch: int - ViT的.patch尺寸（图像分块大小）
        :param dim: int - ViT的特征维度（嵌入维度）
        """
        super(AutoEncoder, self).__init__()
        # 保存模型核心参数（供前向传播使用）
        self.P, self.L, self.size, self.dim = P, L, size, dim

        # 1. 编码器（Encoder）：CNN卷积层，用于提取高光谱图像的光谱-空间特征
        # 输入：(1, L, size, size) → 输出：(1, (dim*P)//patch**2, size, size)
        self.encoder = nn.Sequential(
            # 第1层：1×1卷积（波段维度压缩），输入L维 → 输出128维
            nn.Conv2d(L, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128, momentum=0.9),  # 批量归一化（加速收敛，防止过拟合）
            nn.Dropout(0.25),  # Dropout正则化（随机丢弃25%神经元，防止过拟合）
            nn.LeakyReLU(),  # 激活函数（带泄露的ReLU，避免梯度消失）
            # 第2层：1×1卷积，128维 → 64维
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(),
            # 第3层：1×1卷积，64维 → (dim*P)//patch²维（适配后续ViT输入）
            nn.Conv2d(64, (dim*P)//patch**2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d((dim*P)//patch**2, momentum=0.5),
        )

        # 2. ViT模块（Vision Transformer）：捕获全局光谱-空间依赖关系
        # 输入：编码器输出特征图 → 输出：全局cls嵌入特征（用于丰度预测）
        self.vtrans = transformer.ViT(
            image_size=size, patch_size=patch, dim=(dim*P), depth=2,
            heads=8, mlp_dim=12, pool='cls'
        )
        
        # 3. 上采样层（Upscale）：将ViT输出的嵌入特征映射为空间维度的丰度图
        self.upscale = nn.Sequential(
            nn.Linear(dim, size ** 2),  # 线性层：dim维 → size²维（对应size×size像素）
        )
        
        # 4. 平滑层（Smooth）：优化丰度图空间连续性，施加Softmax约束（丰度和为1）
        self.smooth = nn.Sequential(
            nn.Conv2d(P, P, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # 3×3卷积（保持尺寸不变）
            nn.Softmax(dim=1),  # 通道维度Softmax（确保每个像素的所有端元丰度和为1）
        )

        # 5. 解码器（Decoder）：从丰度图重构高光谱图像
        # 输入：(1, P, size, size) 丰度图 → 输出：(1, L, size, size) 重构高光谱图像
        self.decoder = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=(1, 1), stride=(1, 1), bias=False),  # 1×1卷积（丰度→高光谱重构）
            nn.ReLU(),  # 激活函数（保证重构值非负）
        )

    @staticmethod
    def weights_init(m):
        """
        模型权重初始化方法（静态方法）
        :param m: nn.Module - 模型的单个层（由apply()自动遍历所有层）
        """
        # 对卷积层使用Kaiming正态初始化（适配ReLU/LeakyReLU激活函数，提升收敛速度）
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        """
        模型前向传播（核心：输入高光谱图像，输出预测丰度+重构图像）
        :param x: torch.Tensor - 输入高光谱图像，形状(1, L, size, size)
        :return: tuple - (abu_est: 预测丰度图, re_result: 重构高光谱图像)
        """
        # 步骤1：编码器提取特征
        abu_est = self.encoder(x)
        # 步骤2：ViT捕获全局依赖，输出cls嵌入特征
        cls_emb = self.vtrans(abu_est)
        # 步骤3：调整特征形状，适配后续上采样（1, P, dim）
        cls_emb = cls_emb.view(1, self.P, -1)
        # 步骤4：上采样，将嵌入特征映射为(size, size)空间尺寸的丰度图
        abu_est = self.upscale(cls_emb).view(1, self.P, self.size, self.size)
        # 步骤5：平滑丰度图，施加Softmax约束（保证丰度物理意义）
        abu_est = self.smooth(abu_est)
        # 步骤6：解码器从丰度图重构高光谱图像
        re_result = self.decoder(abu_est)
        
        return abu_est, re_result


# -------------------------
# NonZeroClipper类：模型参数非负约束器
# 作用：确保解码器的卷积层权重非负（符合高光谱解混的物理约束，避免重构值为负）
# -------------------------
class NonZeroClipper(object):
    def __call__(self, module):
        """
        调用方法（由model.apply()自动遍历所有层，施加约束）
        :param module: nn.Module - 模型的单个层
        """
        # 仅对包含weight属性的层施加约束
        if hasattr(module, 'weight'):
            w = module.weight.data  # 获取层的权重数据
            w.clamp_(1e-6, 1)  # 权重裁剪：下限1e-6（避免0），上限1（防止权重过大）


# -------------------------
# Train_test类：模型训练+测试+结果保存主类
# 作用：封装完整的实验流程，包括数据加载、模型训练、评估、可视化、结果保存
# -------------------------
class Train_test:
    def __init__(self, dataset, device, skip_train=False, save=False):
        """
        初始化实验配置与数据集
        :param dataset: str - 数据集名称（samson/apex/dc）
        :param device: torch.device - 计算设备（cuda:0/cpu）
        :param skip_train: bool - 是否跳过训练（直接加载已保存模型）
        :param save: bool - 是否保存模型权重、损失、结果文件
        """
        super(Train_test, self).__init__()
        # 保存实验配置参数
        self.skip_train = skip_train  # 是否跳过训练
        self.device = device          # 计算设备
        self.dataset = dataset        # 数据集名称
        self.save = save              # 是否保存结果

        # 添加光谱平滑性损失权重参数
        self.lambda_smooth = 10  # 平滑性损失的权重，可根据需要调整！！！！！

        # 1. 创建结果保存目录（按数据集命名，避免结果覆盖）
        self.save_dir = "trans_mod_" + dataset + "/"
        os.makedirs(self.save_dir, exist_ok=True)  # 目录已存在则不报错

        # 2. 按数据集配置核心超参数（差异化适配，保证实验效果）
        if dataset == 'samson':
            # Samson数据集：3端元、156波段、95×95像素
            self.P, self.L, self.col = 3, 156, 95
            self.LR, self.EPOCH = 6e-3, 200  # 学习率、训练轮数
            self.patch, self.dim = 5, 200    # ViT.patch尺寸、嵌入维度
            self.beta, self.gamma = 5e3, 3e-2  # 重构损失、SAD损失的权重系数
            self.weight_decay_param = 4e-5   # 优化器权重衰减（正则化）
            self.order_abd, self.order_endmem = (0, 1, 2), (0, 1, 2)  # 丰度/端元顺序调整（对齐真实值）
            # 加载数据集（调用datasets.py的Data类）
            self.data = datasets.Data(dataset, device)
            # 生成数据加载器（批量大小=像素总数，全批次训练）
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            # 加载端元初始权重（用于解码器初始化，适配物理意义）
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()


        elif dataset == 'jasper':
            self.P, self.L, self.col = 4, 198, 100
            self.LR, self.EPOCH = 6e-3, 300 #从 8e-3 降到 4e-3，轮数从 200 增到 300
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 5e3, 6e-2  # 将 gamma 从 4e-2 提高到 6e-2，强制模型更关注光谱形状
            self.weight_decay_param = 3e-5
            # 注意：如果训练出来颜色不对，可以调整这个顺序对齐真值标签
            self.order_abd, self.order_endmem = (0, 1, 2, 3), (0, 1, 2, 3)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()


        elif dataset == 'apex':
            # APEX数据集：4端元、285波段、110×110像素
            self.P, self.L, self.col = 4, 285, 110
            self.LR, self.EPOCH = 8e-3, 200
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 5e3, 5e-2
            self.weight_decay_param = 4e-5
            self.order_abd, self.order_endmem = (0, 1, 2, 3), (0, 1, 2, 3)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()

        elif dataset == 'dc':
            # DC数据集：6端元、191波段、80×80像素
            self.P, self.L, self.col = 6, 191, 80
            self.LR, self.EPOCH = 6e-3, 150
            self.patch, self.dim = 10, 400
            self.beta, self.gamma = 5e3, 1e-4
            self.weight_decay_param = 3e-5
            self.order_abd, self.order_endmem = (0, 2, 1, 5, 4, 3), (0, 2, 1, 5, 4, 3)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()

        else:
            # 未知数据集抛出异常
            raise ValueError("Unknown dataset - 仅支持jasper/samson/apex/dc")

    def run(self, smry):
        """
        执行完整的模型训练+测试流程（核心方法）
        :param smry: bool - 是否打印模型结构摘要（需启用torchsummary）
        """
        # 1. 初始化模型并移至指定设备
        net = AutoEncoder(
            P=self.P, L=self.L, size=self.col,
            patch=self.patch, dim=self.dim
        ).to(self.device)

        # 2. 打印模型结构摘要（若smry=True，需启用torchsummary）
        if smry:
            # summary(net, (1, self.L, self.col, self.col), batch_dim=None)
            return

        # 3. 模型权重初始化（调用AutoEncoder的weights_init方法）
        net.apply(net.weights_init)

        # （注释部分：解码器权重初始化，使用数据集提供的初始端元权重）
        model_dict = net.state_dict()
        model_dict['decoder.0.weight'] = self.init_weight
        net.load_state_dict(model_dict)

        # 4. 配置损失函数、优化器、学习率调度器
        loss_func = nn.MSELoss(reduction='mean')  # 重构损失：均方误差（MSE）
        loss_func2 = utils.SAD(self.L)            # SAD损失：光谱角距离（高光谱解混专属指标）
        # 优化器：Adam（自适应学习率，收敛稳定）
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.LR, weight_decay=self.weight_decay_param
        )
       # 4. 配置优化器后的调度器修改
        # 原来是 step_size=15, gamma=0.8 (衰减太快)
        # 建议改为：每 50 轮减半，配合 300 轮的总轮数
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5
        )

        # 实例化参数非负约束器
        apply_clamp_inst1 = NonZeroClipper()
        
        # 5. 模型训练（若不跳过训练）
        if not self.skip_train:
            time_start = time.time()  # 记录训练开始时间
            net.train()  # 模型切换为训练模式（启用Dropout、BatchNorm训练模式）
            epo_vs_los = []  # 保存每轮训练损失，用于后续可视化

            # 5.1 训练轮数循环
            for epoch in range(self.EPOCH):
                # 5.2 批次循环（全批次训练，仅1个批次）
                for i, (x, _) in enumerate(self.loader):
                    # 调整输入数据形状：适配模型输入（1, L, col, col）
                    x = x.transpose(1, 0).view(1, -1, self.col, self.col)
                    # 前向传播：获取预测丰度和重构图像
                    abu_est, re_result = net(x)

                    # 5.3 计算损失（总损失=重构损失+SAD损失+光谱平滑性损失）
                    loss_re = self.beta * loss_func(re_result, x)  # 重构损失（乘以权重beta）
                    # 计算SAD损失（调整数据形状，适配utils.SAD计算）
                    loss_sad = loss_func2(
                        re_result.view(1, self.L, -1).transpose(1, 2),
                        x.view(1, self.L, -1).transpose(1, 2)
                    )
                    loss_sad = self.gamma * torch.sum(loss_sad).float()  # SAD损失（乘以权重gamma）
                    
                    # 计算光谱平滑性损失
                    est_endmem = net.state_dict()["decoder.0.weight"].cpu().numpy()
                    est_endmem = torch.tensor(est_endmem.reshape((self.L, self.P)), device=self.device)
                    spectral_smoothness_loss = SpectralSmoothnessLoss()(est_endmem)
                    
                    # 总损失
                    total_loss = loss_re + loss_sad + self.lambda_smooth * spectral_smoothness_loss

                    # 5.4 反向传播与参数更新
                    optimizer.zero_grad()  # 清空梯度（避免梯度累积）
                    total_loss.backward()  # 反向传播，计算梯度
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)  # 梯度裁剪（防止梯度爆炸）
                    optimizer.step()  # 优化器更新模型参数

                    # 5.5 施加参数非负约束（仅对解码器生效）
                    net.decoder.apply(apply_clamp_inst1)
                    
                    # 5.6 每10轮打印一次训练日志
                    if epoch % 10 == 0:
                        print('Epoch:', epoch, '| train loss: %.4f' % total_loss.data,
                              '| re loss: %.4f' % loss_re.data,
                              '| sad loss: %.4f' % loss_sad.data,
                              '| smooth loss: %.4f' % (self.lambda_smooth * spectral_smoothness_loss.data))
                    # 保存当前轮次总损失
                    epo_vs_los.append(float(total_loss.data))

                # 5.7 每轮训练结束后，更新学习率
                scheduler.step()

            # 5.8 训练结束，记录耗时并保存结果
            time_end = time.time()
            if self.save:
                # 保存模型权重（pickle格式）
                with open(self.save_dir + 'weights_new.pickle', 'wb') as handle:
                    pickle.dump(net.state_dict(), handle)
                # 保存训练损失（.mat格式，供后续分析）
                sio.savemat(self.save_dir + f"{self.dataset}_losses.mat", {"losses": epo_vs_los})
            
            # 打印训练总耗时
            print('Total computational cost:', time_end - time_start)

        # 6. 加载已保存的模型权重（若跳过训练）
        else:
            with open(self.save_dir + 'weights.pickle', 'rb') as handle:
                net.load_state_dict(pickle.load(handle))

        # 7. 模型测试/评估（切换为评估模式）
        net.eval()  # 模型切换为评估模式（关闭Dropout、固定BatchNorm）
        with torch.no_grad():  # 禁用梯度计算（加速推理，节省显存）
            # 7.1 加载测试数据并前向传播
            x = self.data.get("hs_img").transpose(1, 0).view(1, -1, self.col, self.col)
            abu_est, re_result = net(x)

            # 7.2 丰度图后处理（保证物理意义：每像素丰度和为1）
            abu_est = abu_est / (torch.sum(abu_est, dim=1))
            # 转换为numpy数组（适配后续保存与可视化）：(col, col, P)
            abu_est = abu_est.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            # 加载真实丰度图并转换为numpy数组
            target = torch.reshape(self.data.get("abd_map"), (self.col, self.col, self.P)).cpu().numpy()
            # 加载真实端元与预测端元（从解码器权重中提取）
            true_endmem = self.data.get("end_mem").numpy()
            est_endmem = net.state_dict()["decoder.0.weight"].cpu().numpy()
            est_endmem = est_endmem.reshape((self.L, self.P))

            # 7.3 调整丰度/端元顺序（对齐真实值，保证指标计算准确）
            abu_est = abu_est[:, :, self.order_abd]
            est_endmem = est_endmem[:, self.order_endmem]

            # 7.4 保存预测结果（.mat格式，供后续分析）
            sio.savemat(self.save_dir + f"{self.dataset}_abd_map.mat", {"A_est": abu_est})
            sio.savemat(self.save_dir + f"{self.dataset}_endmem.mat", {"E_est": est_endmem})

            # 7.5 可视化端元光谱（matplotlib绘制，保存为png）
            import matplotlib.pyplot as plt
            import numpy as np
            plt.figure(figsize=(10, 5))
            # 根据数据集类型定义地物类别名称
            if self.dataset == 'samson':
                class_names = ['Soil', 'Tree', 'Water']
            elif self.dataset == 'jasper':
                class_names = ['Veg', 'Soil', 'Water', 'Road']
            else:
                class_names = [f'Endmember {i + 1}' for i in range(self.P)]
                
            for i in range(self.P):
                plt.plot(np.arange(self.L), est_endmem[:, i], label=class_names[i])
            plt.xlabel('Spectral Band')
            plt.ylabel('Reflectance')
            plt.title(f'{self.dataset} Estimated Endmember Spectra')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.save_dir + f"{self.dataset}_endmembers.png", dpi=300)
            plt.close()

            # 7.6 可视化丰度图（子图绘制，保存为png）
            fig, axes = plt.subplots(1, self.P, figsize=(4 * self.P, 4))
            # 根据数据集类型定义地物类别名称
            if self.dataset == 'samson':
                class_names = ['Soil', 'Tree', 'Water']
            elif self.dataset == 'jasper':
                class_names = ['Veg', 'Soil', 'Water', 'Road']
            else:
                class_names = [f'Endmember {i + 1}' for i in range(self.P)]
                
            for i in range(self.P):
                ax = axes[i] if self.P > 1 else axes
                im = ax.imshow(abu_est[:, :, i], cmap='viridis')
                ax.set_title(f'{class_names[i]} Abundance')
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(self.save_dir + f"{self.dataset}_abundance.png", dpi=300)
            plt.close()

            # 7.7 计算并打印评估指标（RE、RMSE、SAD）
            x = x.view(-1, self.col, self.col).permute(1, 2, 0).detach().cpu().numpy()
            re_result = re_result.view(-1, self.col, self.col).permute(1, 2, 0).detach().cpu().numpy()
            re = utils.compute_re(x, re_result)  # 计算重构误差（RE）
            print("RE:", re)

            rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)  # 计算RMSE（丰度误差）
            print("Class-wise RMSE value:")
            for i in range(self.P):
                print("Class", i + 1, ":", rmse_cls[i])
            print("Mean RMSE:", mean_rmse)

            sad_cls, mean_sad = utils.compute_sad(est_endmem, true_endmem)  # 计算SAD（光谱角误差）
            print("Class-wise SAD value:")
            for i in range(self.P):
                print("Class", i + 1, ":", sad_cls[i])
            print("Mean SAD:", mean_sad)

            # 7.8 记录实验日志（csv格式，方便后续对比实验）
            with open(self.save_dir + "log1.csv", 'a') as file:
                file.write(f"LR: {self.LR}, ")
                file.write(f"WD: {self.weight_decay_param}, ")
                file.write(f"RE: {re:.4f}, ")
                file.write(f"SAD: {mean_sad:.4f}, ")
                file.write(f"RMSE: {mean_rmse:.4f}\n")

            # 7.9 调用plots模块，生成更详细的结果可视化图
            plots.plot_abundance(target, abu_est, self.P, self.save_dir, self.dataset)
            plots.plot_endmembers(true_endmem, est_endmem, self.P, self.save_dir, self.dataset)

# =================================================================
# 主入口：该文件作为模块被main.py导入，不直接运行（故pass）
if __name__ == '__main__':
    pass