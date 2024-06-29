import sys
import torch
from torch import nn
from torchstat import stat
import torch.nn.functional as F
from torch import Tensor
from Net.SRM.getResidual import HPF
from Net.MPNCOV import CovpoolLayer, SqrtmLayer
from Net.ISTS_basicBlock import *


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


class RNsteam(nn.Module):
    def __init__(self):
        super(RNsteam, self).__init__()
        self.pre = HPF()
        self.gct1 = GCT(30, mode='l2')
        self.Pre_conv_1 = ConvBn(30, 64, 1, 1, 0, bias=False)
        # Truncation, threshold = 3
        self.TLU = nn.Hardtanh(-3, 3)
        self.gct2 = GCT(64, mode='l2')
        self.conv_1 = ConvBn(64, 32, 1, 1, 0, bias=False)

    def forward(self, inputs):
        out = self.pre(inputs)
        out = self.gct1(out)
        out = self.Pre_conv_1(out)
        out = self.TLU(out)
        out = self.gct2(out)
        out = self.conv_1(out)
        return out


class ICsteam(nn.Module):
    def __init__(self):
        super(ICsteam, self).__init__()
        self.b1_1 = Block1(1, 64)
        self.b1_2 = Block1(64, 16)

        self.b2_1 = Block2(16, 16)
        self.b2_2 = Block2(16, 16)
        self.b2_3 = Block2(16, 16)

    def forward(self, inputs):
        x = self.b1_1(inputs)
        x = self.b1_2(x)

        x = self.b2_1(x)
        x = self.b2_2(x)
        output = self.b2_3(x)
        return output


class DCAM(nn.Module):
    def __init__(self, inchannels1=32, inchannels2=16, outchannels=32):
        super(DCAM, self).__init__()
        # self.att1 = MultiSpectralAttentionLayer(channel=inchannels1, dct_h=inchannels1,
        #                                         reduction=2, dct_w=256)

        self.att0 = MultiSpectralAttentionLayer(channel=inchannels2, dct_h=inchannels2,
                                                reduction=2, dct_w=256)

        self.blk = nn.Sequential(
            nn.Conv2d(inchannels1 + inchannels2, outchannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU()
        )
        self.att2 = MultiSpectralAttentionLayer(channel=outchannels, dct_h=outchannels,
                                                reduction=2, dct_w=256)

    def forward(self, inputs1, inputs2):  # inputs1 noise inputs2 content
        # x_RN = self.att1(inputs1)
        x_IC = self.att0(inputs2)
        x = torch.cat((x_IC, inputs1), dim=1)
        # x = torch.cat((inputs2, inputs1), dim=1)
        x = self.blk(x)
        output = self.att2(x)
        return output


class TS_PRE_GFC(nn.Module):
    def __init__(self, outchannels=32):
        super(TS_PRE_GFC, self).__init__()
        self.ICsteam = ICsteam()
        self.RNsteam = RNsteam()
        self.dcam = DCAM(inchannels1=32, inchannels2=16, outchannels=32)

    def forward(self, inputs):
        x_IC = self.ICsteam(inputs)  # [b, 16, 256, 256]
        x_RN = self.RNsteam(inputs)  # [b, 32, 256, 256]
        # print(x_RN.shape)
        output = self.dcam(x_RN, x_IC)  # [b, 32, 256, 256]
        return output


class StylePool(nn.Module):
    def __init__(self, eps=1e-5):
        super(StylePool, self).__init__()
        self.eps = eps

    def forward(self, x):
        N, C, _, _ = x.size()

        avgpool = x.view(N, C, -1).mean(dim=2, keepdim=True)
        varpool = x.view(N, C, -1).var(dim=2, keepdim=True) + self.eps
        # stdpool = varpool.sqrt()

        # Covpool
        # covpool = CovpoolLayer(x)
        # covpool = SqrtmLayer(covpool, 5)
        # covpool = covpool.view(N, C, -1).mean(dim=2, keepdim=True)

        # t = torch.cat((avgpool, stdpool, covpool), dim=1)
        t = torch.cat((avgpool, varpool), dim=1)
        return t


class ISTSNet(nn.Module):
    def __init__(self):
        super(ISTSNet, self).__init__()
        # Proprecessing
        self.ts_pre = TS_PRE_GFC()

        # Feasture exact
        self.downSample1 = nn.Sequential(
            FEBlock(32, 32),
            FEBlock(32, 32),
            ConAvg(32, 48))
        self.downSample2 = nn.Sequential(
            FEBlock(48, 48),
            FEBlock(48, 48),
            ConAvg(48, 64))
        self.downSample3 = nn.Sequential(
            FEBlock(64, 64),
            FEBlock(64, 64),
            ConAvg(64, 128))

        # Stylepooling
        self.StylePool = StylePool()
        # classifier
        self.fc = nn.Linear(128 * 2, 2)

    def forward(self, input):
        # Proprecessing
        # out = self.pre(input)
        # out = self.gct1(out)
        # out = self.Pre_conv_1(out)
        # out = self.TLU(out)
        # out = self.gct2(out)
        out = self.ts_pre(input)
        # Feasture exact

        out = self.downSample1(out)
        out = self.downSample2(out)
        out = self.downSample3(out)

        # Stylepooling
        out = self.StylePool(out)
        out = out.view(out.size(0), -1)
        # FC
        output = self.fc(out)
        return output


if __name__ == '__main__':
    model = ISTSNet()
    print(model)
    stat(model, (1, 256, 256))
    # x = torch.ones(size=(8, 1, 256, 256)).cuda()
    # print(x.shape)
    #
    # net = ISTSNet().cuda()
    # print(net)
    #
    # output_Y = net(x)
    # print(output_Y.shape)
