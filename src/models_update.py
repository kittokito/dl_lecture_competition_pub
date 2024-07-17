import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# def calculate_padding(kernel_size, stride=1, dilation=1):
#     padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
#     if padding < 0:
#         print(f"Warning: Negative padding calculated: {padding}. Setting to 0.")
#         return 0
#     return padding

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        p_drop: float = 0.5,  # ドロップアウト率を増加
        weight_decay: float = 1e-4  # L2正則化係数
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim, p_drop),
            ConvBlock(hid_dim, hid_dim, p_drop),
            ConvBlock(hid_dim, hid_dim, p_drop),
            ConvBlock(hid_dim, hid_dim, p_drop)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

        self.weight_decay = weight_decay

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)
        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.5,  # ドロップアウト率を増加
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        # padding = calculate_padding(kernel_size)

        # print(f"ConvBlock padding: {padding}")  # デバッグ用出力

        self.conv0 = nn.Conv1d(in_dim, out_dim, int(kernel_size), padding=1)
        self.conv1 = nn.Conv1d(out_dim, out_dim, int(kernel_size), padding=1)
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        
        self.layernorm0 = nn.LayerNorm(out_dim)
        self.layernorm1 = nn.LayerNorm(out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))
        X = self.layernorm0(X)

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))
        X = self.layernorm1(X)

        return self.dropout(X)