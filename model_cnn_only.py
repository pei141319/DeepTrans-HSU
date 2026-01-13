import torch
import torch.nn as nn

class PureCNNUnmixing(nn.Module):
    def __init__(self, n_bands, n_endmembers, img_size=95):
        super(PureCNNUnmixing, self).__init__()
        self.n_bands = n_bands
        self.n_endmembers = n_endmembers

        # --- Encoder: 纯卷积提取丰度 ---
        self.encoder = nn.Sequential(
            # 第一层：捕获空间局部特征
            nn.Conv2d(n_bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 第二层：进一步提取深层特征
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 第三层：1x1 卷积代替 Transformer 映射到丰度空间
            # 在这里直接将 128 维特征压缩到端元数量
            nn.Conv2d(128, n_endmembers, kernel_size=1),
            nn.Softmax(dim=1) # 保证丰度非负且和为1 (ANC/ASC 约束)
        )

        # --- Decoder: 线性重构层 (代表端元光谱) ---
        # 为了实验公平，Decoder 必须与原模型完全一致
        self.decoder = nn.Conv2d(n_endmembers, n_bands, kernel_size=1, bias=False)

    def forward(self, x):
        # x shape: [Batch, Bands, H, W]
        abundance = self.encoder(x)
        reconstructed = self.decoder(abundance)
        return abundance, reconstructed

    # 用于提取端元光谱（即 Decoder 的权重）
    def get_endmembers(self):
        return self.decoder.weight.data.view(self.n_bands, self.n_endmembers)