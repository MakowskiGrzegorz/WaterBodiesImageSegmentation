import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import GANConfig
from discriminator_blocks import _discriminator_blocks


class Discriminator(nn.Module):
    """Some Information about Discriminator"""
    def __init__(self, config:GANConfig):
        super(Discriminator, self).__init__()

        self.input = nn.Sequential(
            nn.Conv2d(config.number_of_channels, config.discriminator_features_number * config.discriminator_features_multipliers[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.main = nn.ModuleList()
        self.main += [_discriminator_blocks[config.discriminator_block_type](config.discriminator_features_number * config.discriminator_features_multipliers[i], config.discriminator_features_number * config.discriminator_features_multipliers[i+1],kernel_size=4, stride=2, padding=1) for i in range(len(config.discriminator_features_multipliers)-1)]
        self.last = nn.Sequential(
            nn.Conv2d(config.discriminator_features_number * config.discriminator_features_multipliers[-1], 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
        self.history = torch.Tensor()
    def forward(self, x,feature=False):
        x = self.input(x)
        for block in self.main:
            x = block(x)
        if feature:
            return x
        x = self.last(x)
        return x
