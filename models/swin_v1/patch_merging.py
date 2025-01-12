"""
This module contains the code for the merge the neighbouring patches.
    * Merge the neightbouting patches by conatenating the 2 * 2 patches it results in 4 * dim
    * After performing the patch merging reduce the dimention of the factor of 2 using the lineat layer

"""

import torch
from torch import nn


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: int, input_dim: int, *args, **kwargs):
        """
        Perform patch merging by downsampling the spatial dimensions of the input
        and concatenating neighboring patches along the channel dimension.

        Args:
            input_resolution (int): input resolution from the previous layer
            input_dim (int): deimension of the given input
        """
        super().__init__(*args, **kwargs)
        self.input_resolution = input_resolution
        self.input_dim = input_dim
        self.reduction = nn.Linear(in_features=self.input_dim * 4, out_features=self.input_dim * 2, bias=True)
        self.norm_layer = nn.LayerNorm(self.input_dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, T, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # get the index from @ every 2 steps
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm_layer(x)
        x = self.reduction(x)

        return x


if __name__ == "__main__":
    # sample input assigned
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_input = torch.rand(32, 3136, 96).to(device)

    # Patch Merging
    model = PatchMerging(input_resolution=(56, 56), input_dim=96).to(device=device)
    print(model(sample_input).shape)
