import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class DiscriminatorBlock(nn.Module):
    """Some Information about DiscriminatorBlock"""
    def __init__(self, features_in, features_out, kernel_size, stride, padding):
        super(DiscriminatorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features_in, features_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(features_out),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class DiscriminatorDropoutBlock(nn.Module):
    """Some Information about DiscriminatorDropoutBlock"""
    def __init__(self, features_in, features_out, kernel_size=3, stride=2, padding=1, bn=True):
        super(DiscriminatorDropoutBlock, self).__init__()
            
        self.block = [nn.Conv2d(features_in, features_out, kernel_size, stride, padding),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Dropout2d(0.25),]
        if bn:
            self.block.append(nn.BatchNorm2d(features_out, 0.8))
        
        self.layer = nn.Sequential(*(self.block))
        
    def forward(self, x):

        x = self.layer(x)
        return x


_discriminator_blocks ={
    'basic' : DiscriminatorBlock,
    'dropout'  : DiscriminatorDropoutBlock
}
