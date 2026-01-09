# -*- coding: utf-8 -*-
"""
transformer.py - 改进型Vision Transformer（ViT）实现（高光谱解混专属）
核心功能：
1. 定义基于「交叉注意力（CrossAttention）」的改进型ViT组件，区别于普通自注意力
2. 实现从补丁嵌入、注意力特征提取到特征聚合的完整流程
3. 为高光谱解混的AutoEncoder模型提供全局光谱-空间特征提取能力
依赖模块：torch（核心）、einops（维度便捷操作）、timm（预定义MLP/DropPath）
"""
# 导入PyTorch核心模块：张量计算、神经网络层
import torch
from torch import nn

# 从timm导入常用组件：DropPath（随机深度，正则化）、Mlp（多层感知机）
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

# 从einops导入维度操作工具：rearrange（维度重排）、repeat（张量复制）、Rearrange（层化维度重排）
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# -------------------------
# 辅助工具函数
# -------------------------
def pair(t):
    """
    辅助函数：将单个值转换为（值, 值）元组，适配图像/补丁的高/宽维度需求
    :param t: int/tuple - 输入的单个尺寸值或已有的尺寸元组
    :return: tuple - 若输入是int则返回(t, t)，若已是tuple则直接返回
    """
    return t if isinstance(t, tuple) else (t, t)


# -------------------------
# 基础组件类：ViT的核心基础模块
# -------------------------
class PreNorm(nn.Module):
    """
    前置归一化（Pre-Normalization）层：ViT的标准组件，在注意力/前馈网络前执行层归一化
    作用：稳定训练过程，加速模型收敛，避免梯度爆炸/消失
    """
    def __init__(self, dim, fn):
        """
        初始化前置归一化层
        :param dim: int - 输入特征的维度（嵌入维度）
        :param fn: nn.Module - 后续要执行的网络层（注意力层/前馈网络）
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 层归一化（适配Transformer的归一化方式）
        self.fn = fn  # 保存后续要执行的网络层（注意力/前馈）

    def forward(self, x, **kwargs):
        """
        前向传播：先归一化，再执行后续网络层
        :param x: torch.Tensor - 输入特征张量，形状通常为(B, N, dim)
        :param kwargs: 传递给后续网络层的额外参数
        :return: torch.Tensor - 后续网络层处理后的输出
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    前馈网络（Feed Forward Network, FFN）：ViT的标准组件，对注意力输出做非线性变换
    结构：两层全连接+GELU激活+Dropout正则化
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        """
        初始化前馈网络
        :param dim: int - 输入/输出特征的维度（嵌入维度）
        :param hidden_dim: int - 中间层的隐藏维度（通常为dim的2-4倍）
        :param dropout: float - Dropout正则化的概率（防止过拟合）
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # 第一层全连接：升维
            nn.GELU(),  # GELU激活函数：非线性变换，比ReLU更平滑
            nn.Dropout(dropout),  # Dropout正则化
            nn.Linear(hidden_dim, dim),  # 第二层全连接：降维，恢复原输入维度
            nn.Dropout(dropout)  # Dropout正则化
        )

    def forward(self, x):
        """
        前向传播：直接执行Sequential封装的网络流程
        :param x: torch.Tensor - 输入特征张量，形状(B, N, dim)
        :return: torch.Tensor - 前向网络处理后的输出，形状(B, N, dim)
        """
        return self.net(x)


# -------------------------
# 核心注意力类：改进型交叉注意力（区别于普通自注意力）
# -------------------------
class CrossAttention(nn.Module):
    """
    交叉注意力（CrossAttention）：改进型注意力机制，以CLS Token为查询（Q），所有补丁Token为键（K）/值（V）
    作用：聚焦全局特征聚合，更高效地捕获高光谱数据的全局光谱-空间依赖关系（优于普通自注意力）
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        初始化交叉注意力层
        :param dim: int - 输入特征的嵌入维度（需被num_heads整除）
        :param num_heads: int - 注意力头数（多头注意力，提升特征表达能力）
        :param qkv_bias: bool - 全连接层是否使用偏置项
        :param qk_scale: float/None - 注意力缩放因子（None则自动计算为head_dim**-0.5）
        :param attn_drop: float - 注意力权重的Dropout概率
        :param proj_drop: float - 最终投影层的Dropout概率
        """
        super().__init__()
        self.num_heads = num_heads  # 保存注意力头数
        # 校验：嵌入维度必须能被注意力头数整除，保证每个头的维度一致
        assert dim % num_heads == 0, f"Dim should be divisible by heads dim={dim}, heads={num_heads}"
        
        head_dim = dim // num_heads  # 单个注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 注意力缩放因子（防止内积过大）

        # 定义Q/K/V的全连接投影层（仅Q针对CLS Token，K/V针对所有Token）
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)  # Q投影层（查询）
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)  # K投影层（键）
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)  # V投影层（值）
        
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力权重Dropout
        self.proj = nn.Linear(dim, dim)  # 注意力输出的最终投影层（恢复维度）
        self.proj_drop = nn.Dropout(proj_drop)  # 投影层输出Dropout

    def forward(self, x):
        """
        前向传播：执行交叉注意力计算（CLS Token查询，所有Token键/值）
        :param x: torch.Tensor - 输入特征张量，形状(B, N, dim)（B=批次，N=Token数=1+num_patches，dim=嵌入维度）
        :return: torch.Tensor - 交叉注意力输出，形状(B, 1, dim)（仅返回CLS Token的特征更新结果）
        """
        B, N, C = x.shape  # 解析输入形状：B=批次大小，N=Token总数，C=嵌入维度

        # 步骤1：生成Q（仅用第一个Token，即CLS Token）
        # 形状变化：(B, 1, C) → (B, 1, num_heads, head_dim) → (B, num_heads, 1, head_dim)
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # 步骤2：生成K（所有Token）
        # 形状变化：(B, N, C) → (B, N, num_heads, head_dim) → (B, num_heads, N, head_dim)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # 步骤3：生成V（所有Token）
        # 形状变化：(B, N, C) → (B, N, num_heads, head_dim) → (B, num_heads, N, head_dim)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 步骤4：计算注意力权重（Q@K^T * 缩放因子）
        # 形状变化：(B, num_heads, 1, head_dim) @ (B, num_heads, head_dim, N) → (B, num_heads, 1, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 最后一维归一化，得到注意力权重
        attn = self.attn_drop(attn)  # 注意力权重Dropout

        # 步骤5：注意力权重加权V，得到注意力输出
        # 形状变化：(B, num_heads, 1, N) @ (B, num_heads, N, head_dim) → (B, num_heads, 1, head_dim)
        # 转置+重塑恢复维度：→ (B, 1, num_heads, head_dim) → (B, 1, C)
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        
        # 步骤6：最终投影+Dropout，返回CLS Token的更新结果
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    """
    交叉注意力块：封装「交叉注意力+可选MLP」的完整模块，带残差连接和随机深度
    作用：构成Transformer的基本层，实现特征的全局提取与非线性变换
    """
    def __init__(self, dim, num_heads, mlp_ratio=3., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.15, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
        """
        初始化交叉注意力块
        :param dim: int - 嵌入维度
        :param num_heads: int - 注意力头数
        :param mlp_ratio: float - MLP隐藏层维度与输入维度的比值
        :param qkv_bias: bool - 注意力层Q/K/V是否用偏置
        :param qk_scale: float/None - 注意力缩放因子
        :param drop: float - 整体Dropout概率
        :param attn_drop: float - 注意力Dropout概率
        :param drop_path: float - 随机深度（DropPath）概率
        :param act_layer: nn.Module - 激活函数（默认GELU）
        :param norm_layer: nn.Module - 归一化层（默认LayerNorm）
        :param has_mlp: bool - 是否启用后续MLP层（默认False，简化计算）
        """
        super().__init__()
        self.norm1 = norm_layer(dim)  # 注意力前置归一化（未实际使用，保留扩展）
        # 实例化交叉注意力层
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # 随机深度（DropPath）：训练时随机丢弃部分层的输出，防止过拟合（无效果则退化为Identity）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp  # 是否启用MLP层标记
        
        # 若启用MLP，初始化MLP层及对应的归一化层
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)  # 计算MLP隐藏层维度
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        前向传播：交叉注意力+残差连接+可选MLP
        :param x: torch.Tensor - 输入特征张量，形状(B, N, dim)
        :return: torch.Tensor - 注意力块输出，形状(B, 1, dim)（仅CLS Token）
        """
        # 残差连接：原始CLS Token + 交叉注意力输出（带随机深度）
        # 注：作者注释中保留了另一种方案，当前方案效果更优
        x = x[:, 0:1, ...] + self.drop_path(self.attn(x))
        
        # 若启用MLP，执行MLP变换+残差连接
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


# -------------------------
# Transformer主干网络：堆叠交叉注意力块+前馈网络
# -------------------------
class Transformer(nn.Module):
    """
    Transformer主干网络：堆叠多层「交叉注意力块+前馈网络」，实现特征的深度提取
    作用：对ViT的Token特征进行多轮全局聚合与非线性变换
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        """
        初始化Transformer主干
        :param dim: int - 嵌入维度
        :param depth: int - Transformer的深度（堆叠的注意力块+前馈网络层数）
        :param heads: int - 注意力头数
        :param dim_head: int - 单个注意力头的维度（未实际使用，保留扩展）
        :param mlp_dim: int - 前馈网络的隐藏层维度
        :param dropout: float - Dropout概率
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # Token特征归一化层
        self.layers = nn.ModuleList([])  # 存储堆叠的网络层
        
        # 堆叠depth层「交叉注意力块+前馈网络」
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 前置归一化+交叉注意力块
                PreNorm(dim, CrossAttentionBlock(dim, num_heads=heads, drop=dropout)),
                # 前置归一化+前馈网络
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """
        前向传播：多轮交叉注意力+前馈网络处理
        :param x: torch.Tensor - 输入特征张量，形状(B, N, dim)
        :return: torch.Tensor - Transformer输出，形状(B, N, dim)
        """
        # 遍历每一层的注意力块和前馈网络
        for attn, ff in self.layers:
            # 步骤1：交叉注意力处理，拼接CLS Token与归一化后的其他Token
            x = torch.cat((attn(x), self.norm(x[:, 1:, :])), dim=1)
            # 步骤2：前馈网络处理+残差连接
            x = ff(x) + x
        
        return x


# -------------------------
# ViT主类：完整的改进型Vision Transformer（封装补丁嵌入+Transformer+特征聚合）
# -------------------------
class ViT(nn.Module):
    """
    改进型ViT主类：基于交叉注意力，适配高光谱数据的特征提取
    作用：接收高光谱图像特征图，输出全局聚合的CLS Token特征（供AutoEncoder后续处理）
    """
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        """
        初始化改进型ViT
        :param image_size: int/tuple - 输入图像的尺寸（高光谱图像的空间尺寸，如95→95×95）
        :param patch_size: int/tuple - 补丁的尺寸（将图像划分为补丁的大小，如5→5×5）
        :param dim: int - 补丁嵌入后的维度
        :param depth: int - Transformer的深度
        :param heads: int - 注意力头数
        :param mlp_dim: int - 前馈网络的隐藏层维度
        :param pool: str - 特征聚合方式（'cls'：CLS Token聚合，'mean'：平均聚合）
        :param channels: int - 输入图像的通道数（高光谱中为波段数，默认3兼容普通图像）
        :param dim_head: int - 单个注意力头的维度
        :param dropout: float - Transformer内部Dropout概率
        :param emb_dropout: float - 补丁嵌入后的Dropout概率
        """
        super().__init__()
        # 处理图像/补丁的尺寸，转为（高，宽）元组
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # 校验：图像尺寸必须能被补丁尺寸整除，保证补丁划分无残留
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size. '

        # 计算补丁总数（空间维度划分后的补丁数量）
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # 校验：特征聚合方式仅支持cls或mean
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 补丁嵌入层：将高维图像特征转换为补丁Token（仅做维度重排，不做线性投影，适配高光谱数据）
        self.to_patch_embedding = nn.Sequential(
            # 维度重排：(B, C, H, W) → (B, num_patches, patch_height*patch_width*C)
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            # 注释：作者未使用线性投影层，直接保留维度重排结果，简化计算且更适配高光谱数据
            # nn.Linear(patch_dim, dim),
        )

        # 位置嵌入：可学习的位置信息，添加到Token中（1, num_patches+1, dim）
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # CLS Token：用于全局特征聚合的特殊Token（1, 1, dim）
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)  # 嵌入后的Dropout正则化

        # 实例化Transformer主干网络
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool  # 特征聚合方式
        self.to_latent = nn.Identity()  # 潜在特征转换（无操作，保留扩展）

    def forward(self, img):
        """
        前向传播：完整的ViT特征提取流程（补丁嵌入→CLS Token拼接→位置嵌入→Transformer→特征聚合）
        :param img: torch.Tensor - 输入高光谱特征图，形状(B, C, H, W)（B=批次，C=波段数，H/W=空间尺寸）
        :return: torch.Tensor - 全局聚合特征，形状(B, dim)（CLS Token/平均聚合结果）
        """
        # 步骤1：补丁嵌入，将图像转为补丁Token
        # 形状变化：(B, C, H, W) → (B, num_patches, patch_dim)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape  # 解析形状：b=批次，n=补丁总数

        # 步骤2：拼接CLS Token（每个批次复制CLS Token，再与补丁Token拼接）
        # 形状变化：CLS Token (1,1,dim) → (b,1,dim)；拼接后 (b, n+1, dim)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # 步骤3：添加位置嵌入（广播适配批次，仅取前n+1个位置嵌入）
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)  # 嵌入后Dropout

        # 步骤4：Transformer主干网络处理，提取全局特征
        x = self.transformer(x)

        # 步骤5：特征聚合（按指定方式提取全局特征）
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # 'mean'：所有Token平均；'cls'：取CLS Token
        x = self.to_latent(x)  # 潜在特征转换（无操作，保留扩展）
        
        return x