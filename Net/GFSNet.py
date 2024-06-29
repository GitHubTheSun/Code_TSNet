import sys
import torch
from torch import nn
from torchstat import stat
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torchsummary import summary

import numpy as np

filter_class_1 = [
    np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 1],
        [0, -1, 0],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [1, -1, 0],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [0, -1, 1],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [0, -1, 0],
        [1, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 1, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
]

filter_class_2 = [
    np.array([
        [1, 0, 0],
        [0, -2, 0],
        [0, 0, 1]
    ], dtype=np.float32),
    np.array([
        [0, 1, 0],
        [0, -2, 0],
        [0, 1, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 1],
        [0, -2, 0],
        [1, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [1, -2, 1],
        [0, 0, 0]
    ], dtype=np.float32),
]

filter_class_3 = [
    np.array([
        [-1, 0, 0, 0, 0],
        [0, 3, 0, 0, 0],
        [0, 0, -3, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, -1, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, -3, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, -1],
        [0, 0, 0, 3, 0],
        [0, 0, -3, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -3, 3, -1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, -3, 0, 0],
        [0, 0, 0, 3, 0],
        [0, 0, 0, 0, -1]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, -3, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, -1, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, -3, 0, 0],
        [0, 3, 0, 0, 0],
        [-1, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [-1, 3, -3, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32)
]

filter_edge_3x3 = [
    np.array([
        [-1, 2, -1],
        [2, -4, 2],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 2, -1],
        [0, -4, 2],
        [0, 2, -1]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [2, -4, 2],
        [-1, 2, -1]
    ], dtype=np.float32),
    np.array([
        [-1, 2, 0],
        [2, -4, 0],
        [-1, 2, 0]
    ], dtype=np.float32),
]

filter_edge_5x5 = [
    np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, -2, 2, -1],
        [0, 0, 8, -6, 2],
        [0, 0, -12, 8, -2],
        [0, 0, 8, -6, 2],
        [0, 0, -2, 2, -1]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1]
    ], dtype=np.float32),
    np.array([
        [-1, 2, -2, 0, 0],
        [2, -6, 8, 0, 0],
        [-2, 8, -12, 0, 0],
        [2, -6, 8, 0, 0],
        [-1, 2, -2, 0, 0]
    ], dtype=np.float32),
]

square_3x3 = np.array([
    [-1, 2, -1],
    [2, -4, 2],
    [-1, 2, -1]
], dtype=np.float32)

square_5x5 = np.array([
    [-1, 2, -2, 2, -1],
    [2, -6, 8, -6, 2],
    [-2, 8, -12, 8, -2],
    [2, -6, 8, -6, 2],
    [-1, 2, -2, 2, -1]
], dtype=np.float32)

all_hpf_list = filter_class_1 + filter_class_2 + filter_class_3 + filter_edge_3x3 + filter_edge_5x5 + [square_3x3,
                                                                                                       square_5x5]

hpf_3x3_list = filter_class_1 + filter_class_2 + filter_edge_3x3 + [square_3x3]
hpf_5x5_list = filter_edge_5x5 + [square_5x5]

normalized_filter_class_2 = [hpf / 2 for hpf in filter_class_2]
normalized_filter_class_3 = [hpf / 3 for hpf in filter_class_3]
# print(normalized_filter_class_3)
normalized_filter_edge_3x3 = [hpf / 4 for hpf in filter_edge_3x3]
normalized_square_3x3 = square_3x3 / 4
normalized_filter_edge_5x5 = [hpf / 12 for hpf in filter_edge_5x5]
normalized_square_5x5 = square_5x5 / 12

all_normalized_hpf_list = filter_class_1 + normalized_filter_class_2 + normalized_filter_class_3 + \
                          normalized_filter_edge_3x3 + normalized_filter_edge_5x5 + [normalized_square_3x3,
                                                                                     normalized_square_5x5]
# print(all_normalized_hpf_list)

normalized_hpf_3x3_list = filter_class_1 + normalized_filter_class_2 + normalized_filter_edge_3x3 + [
    normalized_square_3x3]  # 8 + 4 + 4 + 1 = 17
# print(normalized_hpf_3x3_list)
normalized_hpf_5x5_list = normalized_filter_class_3 + normalized_filter_edge_5x5 + [
    normalized_square_5x5]  # 4 + 1 + 8 = 13
# print(normalized_hpf_5x5_list)


normalized_3x3_list = normalized_filter_edge_3x3 + [normalized_square_3x3]
normalized_5x5_list = normalized_filter_edge_5x5 + [normalized_square_5x5]


class GCT(nn.Module):
    '''
    Zongxin Yang, Linchao Zhu, Y. Wu, and Y. Yang, “Gated Channel Transformation for Visual Recognition,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA: IEEE, Jun. 2020, pp. 11791–11800.
    doi: 10.1109/CVPR42600.2020.01181.
    '''

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')
            sys.exit()

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()
        all_hpf_list_3x3 = np.array(normalized_hpf_3x3_list)
        all_hpf_list_5x5 = np.array(normalized_hpf_5x5_list)
        hpf_weight_3x3 = nn.Parameter(torch.Tensor(all_hpf_list_3x3).view(17, 1, 3, 3), requires_grad=False)
        hpf_weight_5x5 = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(13, 1, 5, 5), requires_grad=False)
        self.hpf_3x3 = nn.Conv2d(1, 17, kernel_size=3, padding=1, bias=False)
        self.hpf_3x3.weight = hpf_weight_3x3
        self.hpf_5x5 = nn.Conv2d(1, 13, kernel_size=5, padding=2, bias=False)
        self.hpf_5x5.weight = hpf_weight_5x5

    def forward(self, input):
        out1 = self.hpf_3x3(input)
        out2 = self.hpf_5x5(input)
        output = torch.cat([out1, out2], 1)
        return output


class Style_Pool(nn.Module):
    def __init__(self, eps=1e-5):
        super(Style_Pool, self).__init__()
        self.eps = eps

    def forward(self, x):
        N, C, _, _ = x.size()

        avgpool = x.view(N, C, -1).mean(dim=2, keepdim=True)
        varpool = x.view(N, C, -1).var(dim=2, keepdim=True) + self.eps
        stdpool = varpool.sqrt()
        t = torch.cat((avgpool, stdpool), dim=1)
        return t


class Partial_conv3(nn.Module):
    '''
    Jierun Chen et al., “Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks.” arXiv, Mar. 07, 2023. doi: 10.48550/arXiv.2303.03667.
    '''

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class FasterNet_Block(nn.Module):
    def __init__(self, in_channels: int):
        super(FasterNet_Block, self).__init__()
        self.Pconv = Partial_conv3(in_channels, 4, forward='split_cat')
        self.Pwconv1 = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(2 * in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.Pwconv2 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, input):
        x = input
        out = self.Pconv(x)
        out = self.Pwconv1(out)
        out = self.relu1(self.bn1(out))
        out = self.bn2(self.Pwconv2(out))
        output = x + out
        output = self.relu2(output)
        return output


class Convbn(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernels: [int, tuple], stride: int, padding: int,
                 bias: bool):
        super(Convbn, self).__init__()
        self.Conv = nn.Conv2d(in_channels, out_channels, kernels, stride, padding, dilation=1, groups=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.Conv(input)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ResNet_block(nn.Module):
    def __init__(self, num_input: int, num_output: int, stride):
        super(ResNet_block, self).__init__()
        self.stride = stride
        self.residual = nn.Sequential(
            Convbn(in_channels=num_input, out_channels=num_output, kernels=3, stride=self.stride, padding=1, bias=True),
            nn.Conv2d(num_output, num_output, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_output),
            )
        self.expand = nn.Sequential(
            nn.Conv2d(num_input, num_output, kernel_size=1, stride=self.stride),
            nn.BatchNorm2d(num_output)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = input
        out = self.residual(x)
        if self.stride == 1 and x.shape == out.shape:
            out += x
        else:
            out = self.expand(x) + out
        out = self.relu(out)
        return out


class GFS_Net(nn.Module):
    def __init__(self):
        super(GFS_Net, self).__init__()
        # Proprecessing
        self.pre = HPF()
        self.gct1 = GCT(30, mode='l2')
        self.Pre_conv_1 = Convbn(30, 64, 1, 1, 0, bias=False)
        # Truncation, threshold = 3
        self.TLU = nn.Hardtanh(-3, 3)
        self.gct2 = GCT(64, mode='l2')
        # Feasture exact
        self.conv_1 = Convbn(64, 32, 1, 1, 0, bias=False)
        self.ResNet1 = ResNet_block(32, 32, 1)
        self.ResNet2 = ResNet_block(32, 32, 1)
        self.conv_3_1 = Convbn(32, 16, 3, 1, 1, bias=True)
        self.downsample1 = nn.AvgPool2d(3, 2, 1)
        self.ResNet3 = ResNet_block(16, 16, 1)
        self.conv_3_2 = Convbn(16, 32, 3, 1, 1, bias=True)
        self.downsample2 = nn.AvgPool2d(3, 2, 1)
        self.ResNet4 = ResNet_block(32, 32, 1)
        self.conv_3_3 = Convbn(32, 64, 3, 1, 1, bias=True)
        self.downsample3 = nn.AvgPool2d(3, 2, 1)
        self.PConv1 = FasterNet_Block(64)
        self.PConv2 = FasterNet_Block(64)
        # Stylepooling
        self.StylePool = Style_Pool()
        # classifier
        self.fc = nn.Linear(64 * 2, 2)

    def forward(self, input):
        # Proprecessing
        out = self.pre(input)
        out = self.gct1(out)
        out = self.Pre_conv_1(out)
        out = self.TLU(out)
        out = self.gct2(out)
        # Feasture exact
        out = self.conv_1(out)
        out = self.ResNet1(out)
        out = self.ResNet2(out)
        out = self.conv_3_1(out)
        out = self.downsample1(out)
        out = self.ResNet3(out)
        out = self.conv_3_2(out)
        out = self.downsample2(out)
        out = self.ResNet4(out)
        out = self.conv_3_3(out)
        out = self.downsample3(out)
        out = self.PConv1(out)
        out = self.PConv2(out)
        # Stylepooling
        out = self.StylePool(out)
        out = out.view(out.size(0), -1)
        # FC
        output = self.fc(out)
        return output


if __name__ == '__main__':

    model = GFS_Net()
    print(model)
    stat(model, (1, 256, 256))
