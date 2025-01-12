"""This module contains the transformer block"""

import torch
from torch import nn
from models.swin_v1.swin_transformer_block.window_attention import WindowMSA
from models.swin_v1.swin_transformer_block.mlp import Mlp


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        window_size: int,
        input_dim: int,
        num_head: int,
        depth: int,
        mlp_ratio: float,
        drop: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.input_dim = input_dim
        self.num_head = num_head
        self.depth = depth
        self.drop = drop

        self.blocks = nn.ModuleList(
            [
                WindowMSA(
                    input_resolution=self.input_resolution,
                    input_dim=self.input_dim,
                    window_size=self.window_size,
                    num_head=self.num_head,
                    shift_size=0 if i % 2 == 0 else self.window_size[0] // 2,
                )
                for i in range(self.depth)
            ]
        )

        self.norm_layer_1 = nn.LayerNorm(self.input_dim)
        self.norm_layer_2 = nn.LayerNorm(self.input_dim)
        mlp_hidden_dim = int(self.input_dim * mlp_ratio)
        self.mlp = Mlp(in_features=self.input_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=self.drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        for blk in self.blocks:
            ln = self.norm_layer_1(x)
            x = blk(ln)
            x = x + ln
            x = self.norm_layer_2(x)
            x = self.mlp(x)

        return x


if __name__ == "__main__":
    # sample input assigned
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_input = torch.rand(32, 784, 192).to(device)
    model = SwinTransformerBlock(
        input_resolution=(28, 28), window_size=(7, 7), input_dim=192, num_head=12, depth=2, mlp_ratio=0.0, drop=0.0
    ).to(device=device)
    print(model(sample_input).shape)
