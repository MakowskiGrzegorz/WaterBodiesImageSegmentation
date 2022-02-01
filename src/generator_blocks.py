import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvTransposeBlock(nn.Module):
    """Basic Generator block. This is a stack of ConvTranspose, Batchnorm and Relu as activation"""
    def __init__(self, features_in, features_out, kernel_size=4, stride=2, padding=1):
        super(ConvTransposeBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(features_in, features_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(features_out),
            nn.ReLU(True)
        )
    def forward(self, x):
        x = self.block(x)
        return x


class UpsampleBlock(nn.Module):
    """Basic Generator block but it rather uses upsample+conv2d rather than conv2dtranspose followed by batchnorm2d and leaky relu"""
    def __init__(self,  features_in, features_out, kernel_size=3, stride=1, padding=1, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(features_in, features_out, kernel_size=kernel_size, stride=stride,padding=padding),
            nn.BatchNorm2d(features_out),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        x = self.block(x)
        return x



_generator_blocks ={
    'transpose' : ConvTransposeBlock,
    'upsample'  : UpsampleBlock
}
