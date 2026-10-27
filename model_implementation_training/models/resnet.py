import torch
import torch.nn.functional as F
from torch import nn

import math
import random
import os
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

# Build Residual blocks

def downsample(in_dims:int, out_dims:int, stride=1):
    """
    Shortcut: Condense features. Change the channel size (from in channels -> out channels).
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_dims, out_channels=out_dims, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_dims)
    )

class BasicResidualBlock(nn.Module):
    """
        Implementation for bsaic block using in resnet18 and resnet34.
    """
    
    expansion = 1
    
    def __init__(self, 
                 in_dims:int, # input channels
                 out_dims:int, # base channel width used inside the block (hidden channels)
                 stride=1, # first stride
                 downsample: nn.Module | None = None, # module transforms the shortcut to match the residual
                 ):
        
        super(BasicResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_dims, out_channels=out_dims, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_dims, out_channels=out_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dims),
        )

        self.identity_block = (
            downsample if downsample is not None else nn.Identity()
        )

    def forward(self, x):
        residual = self.block(x)
        identity = self.identity_block(x)

        return F.relu(residual + identity) 

class BottleneckResidualBlock(nn.Module):
    """
        Implementation for bottleneck block using in resnet50+.
    """
    
    expansion = 4

    def __init__(self, 
                 in_dims:int, # input channels
                 out_dims:int, # base channel width used inside the block (hidden channels)
                 stride:int=1, # first stride
                 downsample: nn.Module | None = None, # module transforms the shortcut to match the residual
                 ):
        
        super(BottleneckResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_dims, out_channels=out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_dims, out_channels=out_dims, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_dims, out_channels=out_dims * self.expansion, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dims * self.expansion),
        )

        self.identity_block = (
            downsample if downsample is not None else nn.Identity()
        )

    def forward(self, x):
        residual = self.block(x)
        identity = self.identity_block(x)
    
        return F.relu(residual + identity) 

# Replicating the Resnet paper (setting for Imagenet 1k)
class Resnet(nn.Module):
    def __init__(self, 
                 Block: type[BasicResidualBlock] | type[BottleneckResidualBlock],
                 layers: Sequence[int],
                 in_channels: int = 3, # color RGB images
                 num_classes=1000,
                 ):
        super(Resnet, self).__init__()

        self.in_dims = 64

        # ImageNet stem (7×7, 64, stride 2 + maxpool)
        # Conv -> BN -> Relu -> maxpool
        # ImageNet stem: 224x224 -> 112x112 -> 56x56.
        # New out size = abs((H - K + 2P)/S) + 1 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.big_layers = nn.Sequential(
            # Based on paper (table 1): spatial size 56 -> 28 -> 14 -> 7
            # stride 1 (aka no size change): spatial size after self.conv1 is the same as the one in first layer
            # stride 2 (aka half size since where applied K = 3, P = 1): latter spatial size = 0.5 * former spatial size
            self._make_layer(Block, 64, layers[0],stride=1),  
            self._make_layer(Block, 128, layers[1], stride=2),
            self._make_layer(Block, 256, layers[2],stride=2),
            self._make_layer(Block, 512, layers[3], stride=2),
        )

        # feature channels = from 64 -> 256 -> 512 -> 1024 -> 2048

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        # classification head
        self.fc = nn.Linear(in_features=512 * Block.expansion, out_features=num_classes)

    def _make_layer(self, 
                    Block: type[BasicResidualBlock] | type[BottleneckResidualBlock], # Block used for inner layers
                    out_dims: int, # base channels
                    number_of_blocks: int,
                    stride:int = 1,
                    ):

        out_channels = out_dims * Block.expansion
        shortcut = None

        if stride != 1 or self.in_dims != out_channels:
            shortcut = downsample(self.in_dims, out_channels, stride)

        layers = [Block(in_dims=self.in_dims, out_dims=out_dims, downsample=shortcut, stride=stride)]

        self.in_dims = out_channels

        for i in range(1, number_of_blocks):
            layers.append(
                Block(in_dims=self.in_dims, out_dims=out_dims)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1 (x) # [N, 3, 224, 224]
        x = self.big_layers(x) # [N, 2048, 7, 7] (spatial: 56 -> 56 -> 28 -> 14 -> 7, feat: 64 -> 256 -> 512 -> 1024 -> 2048)
        x = self.avg_pool(x) # [N, 2048, 1, 1]
        x = torch.flatten(x, 1) # [N, 2048]
        x = self.fc(x) # [N, num_classes]

        return x

if __name__ == "__main__":
    print("RESNET Architecture (example)")
    print("RESNET = img -> stem -> maxpool -> big_layers")
    net = Resnet(BottleneckResidualBlock, layers=[3,4,6,3])
    net = net.to(memory_format=torch.channels_last)
    print(net)