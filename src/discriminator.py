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
    def forward(self, x):
        x = self.input(x)
        for block in self.main:
            x = block(x)
        # if feature:
        #     return x
        x = self.last(x)
        #print("Loss:", x)
        return x

    


# class DiscriminatorNew(nn.Module):
#     """Some Information about DiscriminatorNew"""
#     def __init__(self):
#         super(DiscriminatorNew, self).__init__()
#         def discriminator_block(in_filters, out_filters, bn=True):
#             block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
#             if bn:
#                 block.append(nn.BatchNorm2d(out_filters, 0.8))
#             return block

#         self.model = nn.Sequential(
#             *discriminator_block(3, 16, bn=False),
#             *discriminator_block(16, 32),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#         )

#         ds_size = image_size //2** 4

#         self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
#     def forward(self, x):
#         x = self.model(x)
#         x = x.view(x.shape[0], -1)
#         validity = self.adv_layer(x)
#         return validity


