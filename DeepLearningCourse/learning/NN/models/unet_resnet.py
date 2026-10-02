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

class Block(nn.Module):
    """
    Block of layers per depth.
    """

    def __init__(self, in_channels, out_channels,):

        super(Block, self).__init__()

        self.block = nn.Sequential(
            # good practice: bias = False before BN -> get rid of redundant computational power + not learnable paramaters
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class ResnetEncoder(nn.Module):
    """
    Use pretrained Resnet as an Encoder of Unet.

    (default: best.pt - pretrained Resnet50 + imagenet1k)
    """

    def __init__(self, pretrained_resnet):

        super(ResnetEncoder, self).__init__()

        # stem = conv -> BN -> Relu
        self.stem = pretrained_resnet.conv1[:3]

        # pool = maxpool
        self.pool = pretrained_resnet.conv1[3]

        # reuse pretrained without resetting weights
        self.stages = pretrained_resnet.big_layers

    def forward(self, x):
        x = self.stem(x)

        # collect feats from block
        skip_feats = [x]

        x = self.pool(x)

        for i, stage in enumerate(self.stages):
            x = stage(x)

            if i < len(self.stages) - 1:
                skip_feats.append(x)

        return x, skip_feats

class Decoder(nn.Module):
    """
    Use same Decoder as basic Unet.
    """

    def __init__(self, 
                 bottleneck_channels=2048,
                 skip_channels=(64, 256, 512, 1024), # shallow -> deep: based on the pretrained resnet50
                 decoder_channels=(256, 128, 64, 32) # deep -> shallow
                 ):

        super(Decoder, self).__init__()

        if not decoder_channels or len(skip_channels) != len(decoder_channels):
            raise ValueError("Each decoder stage needs one skip channel count.")

        skip_channels = tuple(reversed(skip_channels))
        channels = (bottleneck_channels, *decoder_channels)
        depth = len(decoder_channels)
        self.out_channels = decoder_channels[-1]

        self.blocks = nn.ModuleList(
            [
                Block(in_channels=channels[i + 1] + skip_channels[i], out_channels=channels[i + 1], )
                for i in range(depth)
            ]
        )

        self.up_convs = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=2, stride=2)
                for i in range(depth)
            ]
        )

    def forward(self, x, skip_features):
        if len(skip_features) != len(self.blocks):
            raise ValueError("Incorrect number of encoder skip features.")
        
        for upconv, block, skip in zip(
            self.up_convs,
            self.blocks,
            reversed(skip_features),
        ):
            x = upconv(x)
            x = self.allign_skip_decoder_feat(skip, x)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return x

    def allign_skip_decoder_feat(self, skip_feature, decoder_feature):
        """
        Resize decoder to match skip (preserve the features extracted from the encoder).
        Make sure the encoder skip feature and decoder feature are the same spatial size.
        
        Return alligned decoder (based on skip shape).
        """
        target_size = skip_feature.shape[-2:]
    
        if decoder_feature.shape[-2:] == target_size:
            return decoder_feature

        # if unmatched size: resize the decoder feature to skip feature
        return F.interpolate(
            decoder_feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

class UnetResnet(nn.Module):
    """
    COmbined: Unet with pretrained resnet encoder.
    """

    def __init__(self, pretrained_resnet, num_classes=3):
        
        super(UnetResnet, self).__init__()

        # encoder
        self.encoder = ResnetEncoder(pretrained_resnet=pretrained_resnet)

        # decoder
        self.decoder = Decoder()

        # restore full resolution
        self.final_up = nn.ConvTranspose2d(
            in_channels=self.decoder.out_channels,  # 32
            out_channels=16,
            kernel_size=2,
            stride=2,
        )

        self.final_block = Block(in_channels=16, out_channels=16)

        self.fc = nn.Conv2d(
            in_channels=16,
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        bottleneck, skip_feats = self.encoder(x)
        x = self.decoder(bottleneck, skip_feats)

        x = self.final_up(x)

        if x.shape[-2:] != input_size:
            x = F.interpolate(
                x,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        x = self.final_block(x)

        return self.fc(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from resnet import Resnet, BottleneckResidualBlock

    resnet50_checkpoint_path = "DeepLearningCourse/learning/NN/checkpoints/20260922-145555_resnet50_best.pt"
    layers = [3, 4, 6, 3]
    pretrained_resnet = Resnet(BottleneckResidualBlock, layers=layers, in_channels=3, num_classes= 1000)

    # Load pretrained weights
    checkpoint = torch.load(
        resnet50_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    pretrained_resnet.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    net = UnetResnet(pretrained_resnet, num_classes=3).to(device)
    net.eval()

    with torch.inference_mode():
        for height, width in [(256, 256), (257, 319)]:
            images = torch.randn(1, 3, height, width, device=device)
            logits = net(images)

            assert logits.shape == (1, 3, height, width)
            print(logits.shape)

    print(net)






