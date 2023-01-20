import torch
import torch.nn as nn

# nn 모듈 상속
class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias)]

        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []

        # 1st CBR2d
        layers += [CBR2d(in_channels, out_channels,
                         kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=bias,
                         norm=norm, relu=relu)]

        # 2nd CBR2d
        layers += [CBR2d(in_channels, out_channels,
                         kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=bias,
                         norm=norm, relu=None)]
        # relu=None

        self.resblk = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.resblk(x)
        # input x 와 resblock 통과한 output 더하여 출력(Elementwise Sum)


# PixelShuffle - 고해상도 이미지를 (ry*rx) 채널을 가진 저해상도 서브 픽셀 이미지로
class PixelShuffle(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)
        # input image 의 batch size, channel size, height, width

        x = x.reshape(B, C, H // ry, ry, W // rx, rx)
        # height 를 ry, width 를 rx 로 나눈 값으로 down-sampling

        x.permute(0, 1, 3, 5, 2, 4)

        x.reshape(B, C * ry * rx, H // ry, W // rx)
        # 원본 형태로,

        return x

# PixelUnshuffle - (ry*rx) 채널을 가진 저해상도 서브 픽셀 이미지를 고해상도 이미지로
class PixelUnshuffle(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C // (ry * rx), ry, rx, H, W)

        x = x.permute(0, 1, 4, 2, 5, 3)

        x.reshape(B, C * ry * rx, H // ry, W // rx)
        # 원본 형태로,

        return x

# PixelShuffle vs. PixelUnshuffle --> 서로 transpose operator 관계,















