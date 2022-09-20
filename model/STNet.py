#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class STNet(nn.Module):
    def __init__(self, batch_size=1, w=94, h=24):
        super(STNet, self).__init__()

        self.localization = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(32, 32, kernel_size=5),
                nn.MaxPool2d(3, stride=3),
                nn.ReLU(True)
                )

        self.fc_loc = nn.Sequential(
                nn.Linear(32 * 14 * 2, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
                )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
        self._batch_size = batch_size
        self._in_ch =3
        self._w = w
        self._h = h
        self.base_grid = self.create_base_grid(self._batch_size, self._in_ch, self._h, self._w)

    def linspace_from_neg_one(self,num_steps,dtype=torch.float32,align_corners=True):
        r = torch.linspace(-1, 1, num_steps, dtype=torch.float32)
        r = r * (num_steps - 1) / num_steps
        return r

    def create_base_grid(self, N,C,H,W):
        # https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py
        base_grid = torch.empty((N,H,W,C), dtype=torch.float32)
        base_grid.select(-1,0).copy_(self.linspace_from_neg_one(W, dtype=torch.float32))
        base_grid.select(-1,1).copy_(self.linspace_from_neg_one(H, dtype=torch.float32).unsqueeze_(-1))
        base_grid.select(-1,2).fill_(1)
        return base_grid

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 14 * 2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        N,C,H,W = x.shape
        # if not torch.onnx.is_in_onnx_export():
        # Move tensor to same device where input image is located
        # Doesn't works during onnx export
        if self.base_grid.device != x.device:
            self.base_grid = self.base_grid.to(x.device)
        
        grid = self.base_grid.view(N,H*W,3).bmm(theta.transpose(1,2))
        grid = grid.view(N, H, W, 2)
        
        # x = self.grid_sample(x, grid, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=True)

        # x = self.f32fwd(x, theta)

        return x

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)  # TODO 在 pytorch 1.6.1 中移除: https://github.com/pytorch/pytorch/issues/42218
    def f32fwd(self, x, theta):
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x
