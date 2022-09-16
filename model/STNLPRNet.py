import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPool2dStep(nn.Module):

    def __init__(self, kernel_size, stride, step, stop):
        super(MaxPool2dStep, self).__init__()
        self.step = step
        self.stop = stop
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size, stride=stride)
        )
    def forward(self, x):
        return self.block(x[:, 0:self.stop:self.step, :, :])


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

class STNLPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate, batch_size=1):
        super(STNLPRNet, self).__init__()
        torch.cuda.empty_cache()

        ######### STN #########
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


        ######### LPR #########
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1), # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            # nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
            small_basic_block(ch_in=64, ch_out=128),    # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            MaxPool2dStep(kernel_size=(3, 3), stride=(1, 2), step=2, stop=128),
            small_basic_block(ch_in=64, ch_out=256),   # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),   # *** 11 ***
            nn.BatchNorm2d(num_features=256),   # 12
            nn.ReLU(),
            MaxPool2dStep(kernel_size=(3, 3), stride=(1, 2), step=4, stop=256),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1), # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        )

        self._batch_size = batch_size
        self._in_ch =3
        self._w = 94
        self._h = 24
        self.base_grid = self.create_base_grid(self._batch_size, self._in_ch, self._h, self._w)

    def linspace_from_neg_one(self,num_steps,dtype=torch.float32,align_corners=False):
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
        ######### STN #########
        N,C,H,W = x.shape

        xs = self.localization(x)
        xs = xs.view(-1, 32 * 14 * 2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # if not torch.onnx.is_in_onnx_export():
        # Move tensor to same device where input image is located
        # Doesn't works during onnx export
        if self.base_grid.device != x.device:
            self.base_grid = self.base_grid.to(x.device)
        grid = self.base_grid.view(N,H*W,3).bmm(theta.transpose(1,2))
        grid = grid.view(N, H, W, 2)
        x = F.grid_sample(x, grid, align_corners=True)

        # x = self.f32fwd(x, theta) #TODO : use instead of steps above if gridSample operator available in futur tensorRT versions

        ######### LPR #########
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]: # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits

    #TODO : use instead of replacement steps if gridSample operator available in futur tensorRT versions
    # @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)  # TODO 在 pytorch 1.6.1 中移除: https://github.com/pytorch/pytorch/issues/42218
    # def f32fwd(self, x, theta):
    #     grid = F.affine_grid(theta, x.size(), align_corners=True)
    #     x = F.grid_sample(x, grid, align_corners=True)
    #     return x

def build_stnlprnet(lpr_max_len=11, phase=False, class_num=36, dropout_rate=0.5, batch_size=1):

    Net = STNLPRNet(lpr_max_len, phase, class_num, dropout_rate, batch_size)

    if phase == "train":
        return Net.train()
    else:
        return Net.eval()
