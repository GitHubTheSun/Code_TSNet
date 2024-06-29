import torch.nn as nn
from torchstat import stat


class FEBlock(nn.Module):
    def __init__(self, inchannels=32, outchannels=32):
        super(FEBlock, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(inchannels, inchannels * 4, 3, 1, 1, groups=8),
            nn.BatchNorm2d(inchannels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannels * 4, outchannels, 1, 1, 0),
            nn.BatchNorm2d(outchannels),

            # nn.Conv2d(inchannels, inchannels, 3, 1, 1, groups=8),
            # nn.BatchNorm2d(inchannels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(inchannels, outchannels, 1, 1, 0),
            # nn.BatchNorm2d(outchannels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        output = self.blk(inputs)
        output = self.relu(inputs + output)
        return output


class FEBlock1(nn.Module):
    def __init__(self, inchannels=32, outchannels=32):
        super(FEBlock1, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, 1, 1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannels, outchannels, 3, 1, 1),
            nn.BatchNorm2d(outchannels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        output = self.blk(inputs)
        output = self.relu(inputs + output)
        return output


class ConAvg(nn.Module):
    def __init__(self, inchannels=32, outchannels=32):
        super(ConAvg, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, 3, 1, 1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2, padding=0)
        )

    def forward(self, inputs):
        return self.blk(inputs)


class testnet(nn.Module):
    def __init__(self):
        super(testnet, self).__init__()
        # Feasture exact
        # self.downSample1 = nn.Sequential(
        #     FEBlock(32, 32),
        #     FEBlock(32, 32),
        #     ConAvg(32, 48))
        # self.downSample2 = nn.Sequential(
        #     FEBlock(48, 48),
        #     FEBlock(48, 48),
        #     ConAvg(48, 64))
        # self.downSample3 = nn.Sequential(
        #     FEBlock(64, 64),
        #     FEBlock(64, 64),
        #     ConAvg(64, 128))

        self.downSample1 = nn.Sequential(
            FEBlock(32, 32),
            FEBlock(32, 32),
            ConAvg(32, 64))
        self.downSample2 = nn.Sequential(
            FEBlock(64, 64),
            FEBlock(64, 64),
            ConAvg(64, 128))
        self.downSample3 = nn.Sequential(
            FEBlock(128, 128),
            FEBlock(128, 128),
            ConAvg(128, 128))

    def forward(self, input):
        out = self.downSample1(input)
        out = self.downSample2(out)
        out = self.downSample3(out)
        return out

if __name__ == '__main__':
    model = testnet()
    print(model)
    stat(model, (32, 256, 256))