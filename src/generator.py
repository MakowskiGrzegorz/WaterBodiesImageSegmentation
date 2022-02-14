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





# class GeneratorNew(nn.Module):
#     """Some Information about GeneratorNew"""
#     def __init__(self):
#         super(GeneratorNew, self).__init__()
#         self.init_size = image_size // 4

#         self.input = nn.Sequential(nn.Linear(nz, 128 * self.init_size **2))
#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, 3, stride=1, padding=1),
#             nn.BatchNorm2d(128, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 3, 3, stride=1, padding=1),
#             nn.Tanh(),
#         )

#     def forward(self, x):
#         x = self.input(x)
#         x = x.view(x.shape[0], 128, self.init_size, self.init_size)
#         img = self.conv_blocks(x)
#         return img


