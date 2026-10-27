import math
import os
import random
import time
from datetime import datetime

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd

from typing import Sequence


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, tv_tensors
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TF

# Layer block
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # padding=1 keeps the spatial size through each conv. The 1x1-crop U-Net of the
        # paper shrinks 128px inputs to nothing by depth 4 (128 -> 124 -> 62 -> 58 -> 29
        # -> 25 -> 12 -> 8 -> 4 -> bottleneck needs 4-4=0), so "same" convs it is.
        # BatchNorm was not in the 2015 paper but makes the deeper stacks trainable at lr=1e-3.
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels=64, depth=4):
        super().__init__()

        self.hidden_channels = tuple(base_channels * (2 ** i) for i in range(depth)) # (64, 128, 256, 512)

        channels = (in_channels, *self.hidden_channels) # (in_channels, 64, 128, 256, 512)

        self.blocks = nn.ModuleList([
            Block(in_channels=channels[i], out_channels=channels[i + 1],)
            for i in range(depth)
        ])

        self.down_scale = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_features = [] # gather skipped features for decoder

        for block in self.blocks:
            x = block(x)
            # learnt features from conv block
            skip_features.append(x)
            x = self.down_scale(x)
        return x, skip_features

class Decoder(nn.Module):
    def __init__(self, bottleneck_channels=1024, base_channels=64, depth=4):
        super().__init__()

        skip_channels = tuple(reversed(tuple(base_channels * (2 ** i) for i in range(depth))))

        channels = (bottleneck_channels, *skip_channels,) # (1024, 512, 256, 128, 64)

        # in = channels[i + 1] * 2 because the upconv output is concatenated with the skip,
        # and both carry channels[i + 1] channels.
        self.blocks = nn.ModuleList([
            Block(in_channels=channels[i + 1] * 2, out_channels=channels[i + 1],)
            for i in range(depth)
        ])

        self.up_convs = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=2, stride=2)
                for i in range(depth)
            ]
        )

    def forward(self, x, skip_features):
        for upconv, block, skip in zip(
            self.up_convs,
            self.blocks,
            reversed(skip_features),
        ):

            x = upconv(x)
            encoder_feature = self.copy_and_crop(skip, x)
            x = torch.cat([x, encoder_feature], dim=1)
            x = block(x)

        return x

    def copy_and_crop(self, skip_feature, decoder_feature):
        # With "same" convs and an input divisible by 2 ** depth the sizes already agree;
        # the crop is the fallback for odd input sizes.
        target_size = decoder_feature.shape[-2:]

        if skip_feature.shape[-2:] == target_size:
            return skip_feature

        return TF.center_crop(skip_feature, list(target_size))

class Unet(nn.Module):
    def __init__(self, in_channels, num_classes, base_channel=64, depth=4):
        super().__init__()

        self.encoder = Encoder(in_channels, base_channel, depth)

        encoder_channels = tuple(base_channel * (2 ** i) for i in range(depth))
        bottleneck_channels = base_channel * (2 ** depth) # 1024

        self.bottle_neck = Block(in_channels=encoder_channels[-1], out_channels=bottleneck_channels)

        self.decoder = Decoder(bottleneck_channels, base_channel, depth)

        self.fc = nn.Conv2d(base_channel, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        # The encoder returns (downsampled features, skips) - both are needed.
        x, skip_features = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x, skip_features)
        return self.fc(x)

if __name__ == "__main__":
    print("UNET Architecture (example)")
    net = Unet(in_channels=3, num_classes=3)
    net = net.to(memory_format=torch.channels_last)
    print(net)