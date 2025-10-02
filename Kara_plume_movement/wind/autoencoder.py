from typing import Tuple, List, Type, Dict, Any

import torch
import torch.nn as nn
from mish import Mish
from torch import Tensor
import torch.nn.functional as F

from MyResidualNetwork import MyResNet, MyBasicBlock


class Encoder(nn.Module):
    def __init__(self, in_channels, n_blocks=9, bottleneck=32, **kwargs):
        super().__init__()
        
        self.in_channels = in_channels
        self.n_blocks = n_blocks
        self.bottleneck = bottleneck
        
        self.network = MyResNet(in_channels=self.in_channels, n_blocks=self.n_blocks, bottleneck=self.bottleneck, **kwargs)
        
    def forward(self, x):
        out = self.network.forward(x)
        return out


class Decoder(nn.Module):
    def __init__(self,
                 in_features: int,
                 start_channels: int,
                 finish_channels: int,
                 n_layers: int,
                 expansion_value: int,
                 increase_value: int,
                 H: int, W: int,
                 H_out: int = 40, W_out: int = 120,
                 activation: Type[torch.nn.Module] = Mish,
                ):
        
        super().__init__()
        
        self.in_features = in_features
        self.start_channels = start_channels
        self.finish_channels = finish_channels
        self.n_layers = n_layers
        self.expansion_value = expansion_value
        self.increase_value = increase_value

        self.H = H
        self.W = W
        self.H_out = H_out
        self.W_out = W_out
        
        self.decoder_linear = nn.Sequential(
            torch.nn.Linear(self.in_features, 1024),
            activation(),
            torch.nn.Linear(1024, self.H*self.W*self.start_channels),
            activation()
        )

        self.decoder_conv_layers = nn.ModuleList()
        current_channels = self.start_channels

        for i in range(self.n_layers):
            decoder_conv_layer = nn.Sequential(
                MyBasicBlock(inplanes=current_channels, expansion=self.expansion_value)
            )
            current_channels = int(current_channels * self.expansion_value)
            self.decoder_conv_layers.append(decoder_conv_layer)

        self.decoder_conv_layers.append(nn.Sequential(MyBasicBlock(inplanes=self.finish_channels, final_layer=True)))
        
    def forward(self, x):
        x = self.decoder_linear.forward(x)
        out = x.reshape(-1, self.start_channels, self.H, self.W)

        new_H = self.H
        new_W = self.W
        for i in range(self.n_layers-1):
            out = self.decoder_conv_layers[i].forward(out)
            new_H *= self.increase_value
            new_W *= self.increase_value
            out = F.interpolate(input=out, size=[new_H,new_H], mode='bicubic')
        
        out = self.decoder_conv_layers[-2].forward(out)
        interpolated = F.interpolate(input=out, size=[self.H_out,self.W_out], mode='bicubic')
        out = self.decoder_conv_layers[-1].forward(interpolated)

        return out
    