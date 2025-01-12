"""This module converts the patch of pixels into the embeddings"""

from typing import Union
import torch
from torch import nn


class PatchEmbeddings(nn.Module):
    def __init__(self, image_resolution: tuple, patch_size: Union[int, tuple], in_channels: int, embed_dim: int):
        """
        Convert the input pixels to patches which is emebddings

        Args:
            image_resolution (tuple): size of an image
            patch_size (Union[int, tuple]): patch token size
            in_channels (int): input image channels
            embed_dims (int): dimention of the resulting embddings
        """
        super().__init__()
        self.image_height = image_resolution[0]
        self.image_width = image_resolution[1]
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Patch size
        if isinstance(patch_size, tuple):
            self.patch_size = patch_size
        elif isinstance(patch_size, list):
            self.patch_size = tuple(patch_size)
        else:
            self.patch_size = (patch_size, patch_size)

        self.conv2d = nn.Conv2d(
            in_channels=in_channels, out_channels=self.embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm_layer = nn.LayerNorm(self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(x)
        B, C, H, W = x.shape
        print(H, W)
        x = x.view(B, C, -1).permute(0, 2, 1)  # B * (H * W) *C
        x = self.norm_layer(x)
        return x


if __name__ == "__main__":
    # sample input assigned
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_input = torch.rand(32, 3, 224, 224).to(device)
    print(sample_input.dtype)
    model = PatchEmbeddings(image_resolution=(224, 224), patch_size=4, in_channels=3, embed_dim=96).to(device=device)
    print(model(sample_input).shape)
