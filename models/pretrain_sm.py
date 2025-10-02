import torch.nn as nn
import torchvision.models

import numpy as np
#from models.net import register_model
import sklearn.metrics
import torch.nn.functional as F

class down_sample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down_sample, self).__init__()

        self.do = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.do(x)

class up_sample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_sample, self).__init__()

        self.do = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        return self.do(x)


class UnetGen(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(UnetGen, self).__init__()
        for p in self.parameters():
            p.requires_grad = False
        self.down1 = down_sample(input_nc, 64)
        self.down2 = down_sample(64, 128)
        self.down3 = down_sample(128, 256)
        self.down4 = down_sample(256, 512)
        self.down5 = down_sample(512, 512)
        self.down6 = down_sample(512, 512)
        self.down7 = down_sample(512, 512)
        #  self.down8 = down_sample(512, 512)
        self.up1 = up_sample(512, 512)
        self.up2 = up_sample(1024, 512)
        self.up3 = up_sample(1024, 512)
        self.up4 = up_sample(1024, 256)
        # self.up3 = up_sample(1024, 256)
        self.up5 = up_sample(512, 128)
        self.up6 = up_sample(256, 64)
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(128, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x = self.down7(x6)
        #   x = self.down8(x7)
        x8 = torch.cat((x6, self.up1(x)), dim=1)
        x9 = torch.cat((x5, self.up2(x8)), dim=1)
        x10 = torch.cat((x4, self.up3(x9)), dim=1)
        x11 = torch.cat((x3, self.up4(x10)), dim=1)
        x12 = torch.cat((x2, self.up5(x11)), dim=1)
        x13 = torch.cat((x1, self.up6(x12)), dim=1)
        # x14 = torch.cat((x1, self.up7(x13)), dim=1)

        return self.up7(x13)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

###
from torch.nn.functional import *