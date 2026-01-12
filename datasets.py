# 导入PyTorch数据集核心模块 - 用于构建自定义数据集
import torch.utils.data
# 导入scipy.io - 读取.mat格式的高光谱数据集（核心依赖）
import scipy.io as sio
# 导入torchvision变换模块 - 用于数据增强/预处理（如归一化、翻转等）
import torchvision.transforms as transforms


# -------------------------
# TrainData类：PyTorch标准数据集封装类
# 作用：将高光谱图像（输入）和丰度图（标签）封装为可迭代的数据集，支持数据变换
# -------------------------
class TrainData(torch.utils.data.Dataset):
    def __init__(self, img, target, transform=None, target_transform=None):
        """
        初始化数据集（原作者版本，新增数据变换接口）
        :param img: torch.Tensor - 高光谱图像数据（输入特征）
        :param target: torch.Tensor - 丰度图数据（标签）
        :param transform: Callable/None - 输入图像的变换函数（如归一化、增强）
        :param target_transform: Callable/None - 标签（丰度）的变换函数
        """
        self.img = img.float()  # 高光谱图像转为float32（适配模型计算）
        self.target = target.float()  # 丰度标签转为float32
        self.transform = transform  # 保存输入图像的变换函数
        self.target_transform = target_transform  # 保存标签的变换函数

    def __getitem__(self, index):
        """
        按索引获取单条数据（支持数据变换）
        :param index: int - 数据索引
        :return: tuple - (变换后的图像, 变换后的标签)
        """
        img, target = self.img[index], self.target[index]
        # 若有输入图像变换，执行变换
        if self.transform:
            img = self.transform(img)
        # 若有标签变换，执行变换
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        """返回数据集总长度（像素总数）"""
        return len(self.img)


# -------------------------
# Data类：高光谱数据集核心加载类
# 作用：读取不同高光谱数据集的.mat文件，解析核心数据并生成数据加载器
# -------------------------
class Data:
    def __init__(self, dataset, device):
        super(Data, self).__init__()  # 显式调用父类（object）的初始化（语法冗余但规范）

        # 拼接数据集路径：./data/[数据集名]_dataset.mat
        data_path = "./data/" + dataset + "_dataset.mat"
        
        # 按数据集名称配置核心参数（P=端元数，L=波段数，col=像素维度/特征维度）
        if dataset == 'samson':
            self.P, self.L, self.col = 3, 156, 95    # Samson：3端元、156波段、95×95像素
        elif dataset == 'jasper':
            self.P, self.L, self.col = 4, 198, 100   # Jasper：4端元、198波段、100×100像素
        elif dataset == 'urban':
            self.P, self.L, self.col = 4, 162, 306   # Urban：4端元、162波段、306×306像素
        elif dataset == 'apex':
            self.P, self.L, self.col = 4, 258, 110   # APEX：4端元、258波段、110×110像素
        elif dataset == 'dc':
            self.P, self.L, self.col = 3, 200, 80    # DC数据集：原作者修改为3端元、200波段、80×80像素

        # 读取.mat格式数据集
        data = sio.loadmat(data_path)

        # 这一步是为了兼容jasper文件（她里面叫 GT，你代码要 M）
        if 'GT' in data and 'M' not in data: 
            data['M'] = data['GT'].T
        if 'M1' not in data: 
            data['M1'] = data['M'] # 如果没有初始权重，就用真值初始化

        # 解析核心数据：.T 表示转置（关键！适配模型输入维度）
        self.Y = torch.from_numpy(data['Y'].T).to(device)  # 高光谱数据（转置后适配模型）(像素, 波段)
        self.Y = self.Y.float() / self.Y.float().max()
        self.A = torch.from_numpy(data['A'].T).to(device)  # 丰度图（转置后适配模型）(像素, 端元)
        self.M = torch.from_numpy(data['M'])               # 真实端元光谱(波段, 端元)
        self.M1 = torch.from_numpy(data['M1'])             # 端元初始权重（模型初始化用）(波段, 端元)

    def get(self, typ):
        """
        按类型返回指定数据（便捷接口）
        :param typ: str - 数据类型标识
            'hs_img'：高光谱图像；'abd_map'：丰度图；'end_mem'：端元；'init_weight'：初始权重
        :return: torch.Tensor/np.ndarray - 对应数据
        """
        if typ == "hs_img":
            return self.Y.float()
        elif typ == "abd_map":  # 原作者用elif（语法更严谨）
            return self.A.float()
        elif typ == "end_mem":
            return self.M
        elif typ == "init_weight":
            return self.M1

    def get_loader(self, batch_size=1):
        """
        生成PyTorch DataLoader（批量加载数据）
        :param batch_size: int - 批次大小（默认1）
        :return: torch.utils.data.DataLoader - 数据加载器
        """
        # 实例化数据集：transform传入空的Compose（预留数据变换接口）
        train_dataset = TrainData(img=self.Y, target=self.A, transform=transforms.Compose([]))
        # 构建DataLoader（原作者显式指定参数名，语法更规范）
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        return train_loader



# """
# datasets.py - 高光谱解混数据集加载模块
# 核心功能：
# 1. 定义PyTorch标准Dataset类（TrainData），封装高光谱图像和丰度标签
# 2. 定义Data类，加载.mat格式高光谱数据集，解析端元/丰度/高光谱数据，并生成数据加载器
# 适配数据集：dc（6端元）、samson（3端元）、jasper（4端元）等高光谱解混数据集
# """
# # 导入PyTorch核心库 - 构建数据集、张量计算
# import torch
# # 导入PyTorch数据集核心模块 - 继承Dataset类实现自定义数据集
# import torch.utils.data
# # 导入scipy.io - 读取.mat格式高光谱数据集（核心）
# import scipy.io as sio
# # 导入torchvision变换模块 - 预留数据增强接口（当前代码未使用）
# import torchvision.transforms as transforms

# # -------------------------
# # TrainData类：PyTorch标准数据集封装类
# # 作用：将高光谱图像（输入）和丰度图（标签）封装为PyTorch可迭代的数据集
# # -------------------------
# class TrainData(torch.utils.data.Dataset):
#     def __init__(self, img, target):
#         """
#         初始化数据集
#         :param img: torch.Tensor - 高光谱图像数据（输入特征），形状通常为 [像素数, 波段数]
#         :param target: torch.Tensor - 丰度图数据（标签），形状通常为 [像素数, 端元数]
#         """
#         # 将高光谱图像转为float32类型（适配PyTorch模型计算精度）
#         self.img = img.float()
#         # 将丰度标签转为float32类型
#         self.target = target.float()

#     def __getitem__(self, index):
#         """
#         按索引获取单条数据（PyTorch Dataset必需方法）
#         :param index: int - 数据索引
#         :return: tuple - (单条高光谱图像数据, 单条丰度标签)
#         """
#         return self.img[index], self.target[index]

#     def __len__(self):
#         """
#         返回数据集总长度（PyTorch Dataset必需方法）
#         :return: int - 数据集样本总数（即高光谱图像的像素总数）
#         """
#         return len(self.img)

# # -------------------------
# # Data类：高光谱数据集核心加载类
# # 作用：读取.mat文件，解析高光谱数据/丰度/端元，并生成数据加载器
# # -------------------------
# class Data:
#     def __init__(self, dataset, device):
#         """
#         初始化并加载高光谱数据集
#         :param dataset: str - 数据集名称（如'dc'/'samson'/'jasper'）
#         :param device: torch.device - 数据存放设备（cuda:0/cpu）
#         """
#         # 拼接数据集路径：./data/[数据集名]_dataset.mat（需确保data文件夹下有对应.mat文件）
#         data_path = f"./data/{dataset}_dataset.mat"
#         # 读取.mat格式数据集（核心：加载高光谱结构化数据）
#         data = sio.loadmat(data_path)

#         # 针对dc数据集的参数配置（6端元复杂场景）
#         if dataset == 'dc':
#             self.P = 6    # P：端元数（dc数据集为6类地物/端元）
#             self.L = 191  # L：波段数（dc数据集有191个有效光谱波段）
#             self.col = 80 # col：数据集维度相关参数（如像素维度/特征维度）

#         # 解析.mat文件中的核心数据并转为PyTorch张量，移至指定设备
#         self.Y = torch.from_numpy(data['Y']).to(device)  # Y：高光谱数据立方体（像素数×波段数）
#         self.A = torch.from_numpy(data['A']).to(device)  # A：真实丰度图（像素数×端元数）
#         self.M = torch.from_numpy(data['M'])             # M：真实端元光谱（波段数×端元数）
#         self.M1 = torch.from_numpy(data['M1'])           # M1：端元初始权重（用于模型初始化）

#     def get(self, typ):
#         """
#         根据类型返回指定数据（便捷接口）
#         :param typ: str - 数据类型标识
#             'hs_img'：高光谱图像数据（Y）
#             'abd_map'：丰度图数据（A）
#             'end_mem'：端元光谱数据（M）
#             'init_weight'：端元初始权重（M1）
#         :return: torch.Tensor/np.ndarray - 对应类型的数据
#         """
#         if typ == "hs_img": 
#             return self.Y.float()  # 返回高光谱图像（转为float32）
#         if typ == "abd_map": 
#             return self.A.float()  # 返回丰度图（转为float32）
#         if typ == "end_mem": 
#             return self.M          # 返回端元光谱（numpy数组，按需转张量）
#         if typ == "init_weight": 
#             return self.M1         # 返回端元初始权重（用于模型权重初始化）

#     def get_loader(self, batch_size=1):
#         """
#         生成PyTorch DataLoader（模型训练时批量加载数据）
#         :param batch_size: int - 批次大小（默认1，高光谱解混常使用单批次/全批次训练）
#         :return: torch.utils.data.DataLoader - 数据加载器
#         """
#         # 实例化TrainData数据集，传入高光谱图像（输入）和丰度图（标签）
#         train_dataset = TrainData(img=self.Y, target=self.A)
#         # 生成DataLoader：shuffle=False（高光谱解混需保持像素空间连续性，不打乱）
#         return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)