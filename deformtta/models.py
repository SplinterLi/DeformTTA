from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeDeformNet(nn.Module):
    """Small U-Net style mask-to-flow network used by DeformTTA."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.enc1 = block(in_channels, 32)
        self.enc2 = block(32, 64)
        self.enc3 = block(64, 128)
        self.enc4 = block(128, 256)
        self.dec3 = block(256 + 128, 128)
        self.dec2 = block(128 + 64, 64)
        self.dec1 = block(64 + 32, 32)
        self.flow = nn.Conv2d(32, out_channels, 3, padding=1)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=True), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True), e1], dim=1))
        return torch.tanh(self.flow(d1))


class MultiClassSpatialTransformer(nn.Module):
    """Apply one two-channel displacement field per foreground class."""

    def __init__(self, size: tuple[int, int], num_classes: int) -> None:
        super().__init__()
        h, w = size
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing="ij",
        )
        self.register_buffer("base_grid", torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0))
        self.num_classes = num_classes

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        warped = []
        for class_index in range(self.num_classes):
            image_c = src[:, class_index + 1 : class_index + 2]
            flow_c = flow[:, class_index * 2 : class_index * 2 + 2].permute(0, 2, 3, 1)
            warped_c = F.grid_sample(
                image_c,
                self.base_grid + flow_c,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            warped.append(warped_c)

        anatomy = torch.cat(warped, dim=1)
        background = torch.clamp(1.0 - anatomy.max(dim=1, keepdim=True).values, 0.0, 1.0)
        return torch.cat([background, anatomy], dim=1)

