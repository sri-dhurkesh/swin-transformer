"""This module contains code for the Window and Shifted Window attention"""

import torch
from torch import nn


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowMSA(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        input_dim: int,
        window_size: int,
        num_head: int,
        attn_drop=0.0,
        proj_drop=0.0,
        shift_size=0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_resolution = input_resolution
        self.input_dim = input_dim
        self.window_size = window_size
        self.num_head = num_head
        self.shift_size = shift_size

        self.qkv = nn.Linear(self.input_dim, self.input_dim * 3, bias=True)
        self.proj = nn.Linear(self.input_dim, self.input_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_head)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # masking the data for shifted window attanetion
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size[1]),
                slice(-self.window_size[1], -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        # input shape : (B, T, C)

        initial_B, intial_T, initial_C = x.shape
        H, W = self.input_resolution
        # print("Original Input shape:", x.shape)
        x = x.view(initial_B, H, W, initial_C)
        # print("Unsqueezing Input shape:", x.shape)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partitioning the windows : (num_windows*B, window_size, window_size, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size[0] * self.window_size[1], initial_C)

        B, T, C = x.shape
        # print("Shape after partitioned:", x.shape)

        ##### Creating An W-MSA and Shifted-Window-MSA

        # create a q, k, v values
        x = self.qkv(x).view(B, T, 3, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
        q, k, v = x[0], x[1], x[2]

        # print("Query and Key.T shape:", (q.shape, v.transpose(-2, -1).shape))

        # need to check why they where dividing with q instead of q * k
        d = (C // self.num_head) ** (-0.5)  # represents root of d -> checkout relative position bias in paper.

        q = q * d
        attn = q @ k.transpose(-2, -1)
        print(attn.shape)

        # get the relative position bais
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # print("Relative position bias table:", relative_position_bias.unsqueeze(0).shape)
        attn = attn + relative_position_bias.unsqueeze(0)

        # print("Attn matrix after adding position bias:", attn.shape)
        print(self.shift_size)

        # Adding attention mask
        if self.shift_size <= 0:
            attn = self.softmax(attn)
        else:
            # print("Attn mask shape:", self.attn_mask.unsqueeze(1).unsqueeze(0).shape)
            nW = self.attn_mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_head, T, T) + self.attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_head, T, T)
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        x = window_reverse(x, window_size=self.window_size, H=H, W=W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(initial_B, H * W, C)

        return x


if __name__ == "__main__":
    # sample input assigned
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_input = torch.rand(32, 3136, 96).to(device)
    model = WindowMSA(input_resolution=(56, 56), window_size=(7, 7), input_dim=96, num_head=12, shift_size=3).to(
        device=device
    )
    print(model(sample_input).shape)
