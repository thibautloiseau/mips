import torch
from torch import nn
from meanshift import MeanShiftCluster


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetDown, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.GroupNorm(16, out_size),
            nn.Mish(),
            # nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_size),
            # nn.ReLU(),
            nn.MaxPool2d(2)
          )

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),  # Upsample and then convolution to avoid artifacts
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.GroupNorm(16, out_size),
            nn.Mish(),
            # nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_size),
            # nn.ReLU(),
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(FinalLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        return self.model(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # self.down1 = UNetDown(in_channels, 64)
        # self.down2 = UNetDown(64, 128)
        # self.down3 = UNetDown(128, 256)
        # self.down4 = UNetDown(256, 512)
        # self.down5 = UNetDown(512, 512)
        #
        # self.up1 = UNetUp(512, 512)
        # self.up2 = UNetUp(1024, 256)
        # self.up3 = UNetUp(512, 128)
        # self.up4 = UNetUp(256, 64)
        #
        # self.final = FinalLayer(128, out_channels)

        self.down1 = UNetDown(in_channels, 16)
        self.down2 = UNetDown(16, 32)
        self.down3 = UNetDown(32, 64)
        self.down4 = UNetDown(64, 128)
        self.down5 = UNetDown(128, 128)

        self.up1 = UNetUp(128, 128)
        self.up2 = UNetUp(256, 64)
        self.up3 = UNetUp(128, 32)
        self.up4 = UNetUp(64, 16)

        self.final = FinalLayer(32, out_channels)

        self.meanshift = MeanShiftCluster()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)

        out = self.final(u4, d1)

        with torch.no_grad():
            seg = self.meanshift(out)

        return {'out': out, 'seg': seg}
        # return {'out': out}
