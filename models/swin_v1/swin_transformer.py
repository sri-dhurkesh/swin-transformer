"""This module has the swin trasnformer class that will create a complete swin trasnformer backbone"""

from typing import List, Tuple
import torch
from torch import nn
import sys

print(sys.path)
from patch_embedding import PatchEmbeddings
from patch_merging import PatchMerging
from swin_transformer_block.swin_transformer_block import SwinTransformerBlock


class SwinTransformer(nn.Module):
    def __init__(
        self,
        image_size: tuple,
        patch_size: int,
        in_channels: int,
        depths: List[int],
        num_heads: List[int],
        embed_dim: int,
        window_size: Tuple[int, int],
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.depths = depths
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.window_size = window_size
        self.patches_resolution = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])

        # patch embeddings
        self.patch_embed = PatchEmbeddings(
            image_resolution=self.image_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
        )

        self.blocks = nn.ModuleList()

        for i in range(self.num_layers):
            input_resolution = (
                self.patches_resolution[0] // (2**i),
                self.patches_resolution[1] // (2**i),
            )

            if i > 0:
                # Add Patch Merging for layers after the first
                prev_input_resolution = (
                    self.patches_resolution[0] // (2 ** (i - 1)),
                    self.patches_resolution[1] // (2 ** (i - 1)),
                )
                print("Prev_input_resolution:", prev_input_resolution)

                patch_merge = PatchMerging(
                    input_resolution=prev_input_resolution,
                    input_dim=embed_dim * (2 ** (i - 1)),
                )
                self.blocks.append(patch_merge)

            swin_block = SwinTransformerBlock(
                input_resolution=input_resolution,
                input_dim=embed_dim * (2**i),
                window_size=self.window_size,
                num_head=num_heads[i],
                depth=depths[i],
                mlp_ratio=4.0,  # Commonly used value
                drop=0.1,  # Example dropout value
            )

            self.blocks.append(swin_block)

        # Patch Merging

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        for idx, layer in enumerate(self.blocks):
            print("=" * 100)
            print(idx)
            print(layer)
            print("=" * 100)
            x = layer(x)
            print("Final out shape:", x.shape)
        return x


if __name__ == "__main__":
    from mmengine import Config

    cfg = Config.fromfile("/workspaces/swin-transformer/configs/swin_v1_config.py")
    dataset_cfg = cfg.dataset

    # sample input assigned
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_input = torch.rand(dataset_cfg.batch_size, 3, 224, 224).to(device)
    print(sample_input.shape)

    # intialise the models
    print(cfg.to_dict())

    model = SwinTransformer(**cfg.to_dict()).to(device=device)
    final_out = model(sample_input)
    print(final_out.shape)
