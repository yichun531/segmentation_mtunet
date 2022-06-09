#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import numpy as np
import torch
from torch import nn


class ConvBNReLU(nn.Module):
    def __init__(self,
                 c_in,
                 c_out,
                 kernel_size,
                 stride=1,
                 padding=1, 
                 activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(c_in,
                              c_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, 1),
            ConvBNReLU(cout, cout, 3, 1, 1, activation=False))
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x # skip connection
        x = self.relu(x)
        return x


class U_encoder(nn.Module):
    def __init__(self):
        super(U_encoder, self).__init__()
        self.res1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.res3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.res4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.trans = DoubleConv(512, 1024)
        
    def forward(self, x):
        features = []
        x = self.res1(x)
        features.append(x)  # (224, 224, 64)
        x = self.pool1(x)

        x = self.res2(x)
        features.append(x)  # (112, 112, 128)
        x = self.pool2(x)

        x = self.res3(x)
        features.append(x)  # (56, 56, 256)
        x = self.pool3(x)
        
        x = self.res4(x)
        features.append(x)  # (28, 28, 512)
        x = self.pool4(x)  
        
        x = self.trans(x) # (14, 14, 1024)

        return x, features


class U_decoder(nn.Module):
    def __init__(self):
        super(U_decoder, self).__init__()
        # dilation=1(default): controls the spacing between the kernel points; also known as the à trous algorithm -> 3x3 kernel
        # stride=2: 1 zeros (spaces) between each input pixel
        self.trans1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.res1 = DoubleConv(1024, 512)
        self.trans2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.res2 = DoubleConv(512, 256)
        self.trans3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.res3 = DoubleConv(256, 128)
        self.trans4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.res4 = DoubleConv(128, 64)

    def forward(self, x, feature):

        x = self.trans1(x)  # (14, 14, 1024) -> (28, 28, 512)
        x = torch.cat((feature[3], x), dim=1) # (28, 28, 1024)
        x = self.res1(x) # (28, 28, 512)
        x = self.trans2(x)  # (56, 56, 256)
        x = torch.cat((feature[2], x), dim=1) # (56, 56, 512)
        x = self.res2(x)  # (56, 56, 256)
        x = self.trans3(x)  # (112, 112, 128)
        x = torch.cat((feature[1], x), dim=1) # (112, 112, 256)
        x = self.res3(x)  # (112, 112, 128)
        x = self.trans4(x)  # (224, 224, 64)
        x = torch.cat((feature[0], x), dim=1) # (224, 224, 128)
        x = self.res4(x) # (224, 224, 64)
        return x


class UNet(nn.Module):
    def __init__(self, out_ch=2):
        super(UNet, self).__init__()
        self.encoder = U_encoder()
        self.decoder = U_decoder()
        self.SegmentationHead = nn.Conv2d(64, out_ch, 1) # out_ch = num_class , kernel_size=1

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.encoder(x)  
        x = self.decoder(x, features)
        x = self.SegmentationHead(x)
        return x

