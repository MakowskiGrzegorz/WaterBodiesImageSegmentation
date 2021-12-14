import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, features_in, features_out):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(features_in, features_out,3, padding=1),
            nn.BatchNorm2d(features_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(features_out, features_out,3,padding=1),
            nn.BatchNorm2d(features_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class Downsample(nn.Module):
    
    def __init__(self, features_in, features_out, downsample):
        super().__init__()
        self.double_conv = DoubleConv(features_in, features_out)
        self.downsample_func = nn.MaxPool2d(downsample,downsample)
        
    def forward(self, x):
        x = self.double_conv(x)
        skip_connection = x
        x = self.downsample_func(x)

        return (skip_connection, x)


class Upsample(nn.Module):

    def __init__(self, features_in, features_out, upsample):
        super().__init__()
        self.double_conv = DoubleConv(features_in, features_out)
        self.upsample_func = nn.ConvTranspose2d(features_in, features_out,upsample,upsample)

    def forward(self, x, skip_connection):
        x = self.upsample_func(x)
        x = self.double_conv(torch.cat([x, skip_connection], dim=1))
        return x


class Bottleneck(nn.Module):
    
    def __init__(self, features_in, features_out):
        super().__init__()
        self.double_conv = DoubleConv(features_in, features_out)

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Unet(nn.Module):

    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.features = [16, 32, 64, 128]
        self.input  = Downsample(features_in=channels_in, features_out=self.features[0], downsample=2)
        self.downsample = nn.ModuleList([Downsample(self.features[i], self.features[i+1], 2) for i in range(0,len(self.features)-2)])
        self.bottleneck = Bottleneck(self.features[-2], self.features[-1])
        self.upsample = nn.ModuleList([Upsample(self.features[i], self.features[i-1], 2) for i in range(len(self.features)-1, 0,-1)])
        self.last = nn.Conv2d(self.features[0], channels_out,1,1)

    def forward(self, x):

        skip_connections = []

        skip_conn, x = self.input(x)
        skip_connections.append(skip_conn)
        
        for down in self.downsample:
            skip_conn, x = down(x)
            skip_connections.append(skip_conn)
        
        x = self.bottleneck(x)
        
        for (i, up) in enumerate(self.upsample):
            x = up(x, skip_connections[-i-1])
        x = self.last(x)

        return x
