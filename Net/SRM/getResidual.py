import numpy as np
import torch
import torch.nn as nn

SRM_npy = np.load('C:\\Users\Lenovo\Desktop\ISTSNet\\Net\SRM\\SRM_Kernels.npy')
# print(SRM_npy)
SRM_npy = torch.tensor(SRM_npy).cuda()


class SRM(nn.Module):
    def __init__(self, in_channels=1, out_channels=30, padding=0, SRM_WR=False):
        super(SRM, self).__init__()
        self.Conv_SRM = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 5),
                                  padding=padding, stride=(1, 1))
        self.Conv_SRM_bn = nn.BatchNorm2d(out_channels)

        self.Conv_SRM.weight = torch.nn.Parameter(SRM_npy, requires_grad=SRM_WR)

    def forward(self, x):
        x = self.Conv_SRM(x)
        x = self.Conv_SRM_bn(x)
        return x


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