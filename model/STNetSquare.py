#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class STNetSquare(nn.Module):
    def __init__(self, batch_size=1, w=48, h=48, find_cut_point=False):
        super(STNetSquare, self).__init__()

        self.find_cut_point = find_cut_point

        self.localization = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(32, 32, kernel_size=5),
                nn.MaxPool2d(3, stride=3),
                nn.ReLU(True)
                )

        self.fc_loc = nn.Sequential(
                nn.Linear(32 * 6 * 6, 32),
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

    def transform_image_with_cutpoint(self, x, cut_point):
        batch_size = x.size(0)
        transformed_images = []

        for i in range(batch_size):
            img = x[i]
            # cut_perc = cut_point[i].item()
            cut_perc = 0.5

            height = img.size(1)
            width = img.size(2)
            half = height // 2

            cut = int(cut_perc * height)

            top_part = img[:, :cut, :]
            bottom_part = img[:, cut:, :]

            if cut_perc < 0.5:
                top_part = F.pad(top_part, (0, 0, 0, half - cut), mode='constant', value=0)
                bottom_part = F.interpolate(bottom_part.unsqueeze(0), size=(half, width), mode='bilinear', align_corners=False)
                bottom_part = bottom_part.squeeze(0)
            elif cut_perc > 0.5:
                bottom_part = F.pad(bottom_part, (0, 0, 0, cut - half), mode='constant', value=0)
                top_part = F.interpolate(top_part.unsqueeze(0), size=(half, width), mode='bilinear', align_corners=False)
                top_part = top_part.squeeze(0)
            # if 0.5, keep it like it is


            # Concatenate the two parts horizontally with fitting to 94 (instead 96)
            transformed_img = torch.cat((top_part[:, :, 1:], bottom_part[:, :, :-1]), dim=2)
            transformed_images.append(transformed_img)

        transformed_images = torch.stack(transformed_images)
        return transformed_images
    
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 6 * 6)
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

        if self.find_cut_point:
            x = self.transform_image_with_cutpoint(x, 0.5) 
        return x

    # @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)  # TODO 在 pytorch 1.6.1 中移除: https://github.com/pytorch/pytorch/issues/42218
    # def f32fwd(self, x, theta):
    #     grid = F.affine_grid(theta, x.size(), align_corners=True)
    #     x = F.grid_sample(x, grid, align_corners=True)
    #     return x
