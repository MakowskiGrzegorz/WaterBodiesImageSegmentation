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

        self.input = self.instantinate_input_layer(config)
        self.main = nn.ModuleList()
        self.main += [_discriminator_blocks[config.discriminator_block_type](config.discriminator_features_number * config.discriminator_features_multipliers[i], config.discriminator_features_number * config.discriminator_features_multipliers[i+1],kernel_size=4, stride=2, padding=1) for i in range(len(config.discriminator_features_multipliers)-1)]

        self.last = self.instantinate_last_layer(config)
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
        self.history = torch.Tensor()

    def instantinate_last_layer(self, config):
        layer = []
        if config.discriminator_last_layer_type == "conv":
            layer += [nn.Conv2d(config.discriminator_features_number * config.discriminator_features_multipliers[-1], 1, kernel_size=4, stride=1, padding=0, bias=False)]
        elif config.discriminator_last_layer_type == "linear":
            ds_size = config.image_size //2 ** 4
            layer += [nn.Linear(config.discriminator_features_number * config.discriminator_features_multipliers[-1] * ds_size **2, 1)]
        
        if config.discriminator_last_layer_activation == "sigmoid":
            layer += [nn.Sigmoid()]
        return nn.Sequential(*layer)

    def instantinate_input_layer(self, config):
        layer = []#nn.ModuleList()
        if config.discriminator_input_layer_type == "conv":
            #layer = nn.Sequential(nn.Conv2d(config.number_of_channels, config.discriminator_features_number * config.discriminator_features_multipliers[0], kernel_size=4, stride=2, padding=1),
            #                      nn.LeakyReLU(0.2, inplace=True))
            layer += [nn.Conv2d(config.number_of_channels, config.discriminator_features_number * config.discriminator_features_multipliers[0], kernel_size=4, stride=2, padding=1)]
            layer += [nn.LeakyReLU(0.2, inplace=True)]
        elif config.discriminator_input_layer_type == "dropout":
            layer += [nn.Conv2d(config.number_of_channels, config.discriminator_features_number * config.discriminator_features_multipliers[0],kernel_size=3, stride=2, padding=1)]
            layer += [nn.LeakyReLU(0.2, inplace=True)]
            layer += [nn.Dropout(0.25)]
        return nn.Sequential(*layer)
    
    def forward(self, x):
        x = self.input(x)
        for block in self.main:
            x = block(x)
        # if feature:
        #     return x
        x = self.last(x)
        return x

    


