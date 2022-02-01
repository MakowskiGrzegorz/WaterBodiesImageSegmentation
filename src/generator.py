import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from generator_blocks import _generator_blocks
from config import GANConfig



class Generator(nn.Module):
    """Basic Generator for GAN framework"""
    def __init__(self, config:GANConfig):
        super(Generator, self).__init__()

        self.input = _generator_blocks[config.generator_block_type](config.latent_vector_size, config.generator_features_number * config.generator_features_multipliers[0], kernel_size=4, stride=1, padding=0)
        self.main = nn.ModuleList()
        self.main += [_generator_blocks[config.generator_block_type](config.generator_features_number * config.generator_features_multipliers[i], config.generator_features_number * config.generator_features_multipliers[i+1], kernel_size=4, stride=2, padding=1) for i in range(len(config.generator_features_multipliers) -1)]
        self.main.append(_generator_blocks[config.generator_block_type](config.generator_features_number * config.generator_features_multipliers[-1], config.generator_features_number, kernel_size=4, stride=2, padding=1))
        
        self.last = nn.Sequential(
            nn.ConvTranspose2d(config.generator_features_number, config.number_of_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))

    def forward(self, x):
        #print(x.shape)
        x = self.input(x)
        #print(x.shape)
        for block in self.main:
            x = block(x)
            #print(x.shape)
        
        x = self.last(x)
        #print(x.shape)
        return x