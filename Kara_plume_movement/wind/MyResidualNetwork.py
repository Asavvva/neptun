from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from mish import Mish
from torch import Tensor


class AddCoords(nn.Module):
    """
    https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2)
                            + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


def my_conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return CoordConv(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        padding_mode='zeros'
    )


def my_conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return CoordConv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MyBasicBlock(nn.Module):
    """
    https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18
    """

    def __init__(self,
                 inplanes: int,
                 expansion: int = 1,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 final_layer = False,
                ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("MyBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in MyBasicBlock")
        
        self.expansion = expansion
        planes = inplanes * expansion
        if type(planes) == float:
            planes = int(planes) #int (planes) allows extension < 1
            
        self.conv1 = my_conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.activation = Mish()
        self.conv2 = my_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        
        if stride != 1 or expansion != 1:
            downsample = nn.Sequential(
                my_conv1x1(inplanes, planes, stride),
                norm_layer(planes))
        self.downsample = downsample
        self.stride = stride
        self.final_layer = final_layer
        
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        if self.final_layer is False:
            out += identity
            out = self.activation(out)

        return out


class MyResNet(nn.Module):
    
    def __init__(self,
                 in_channels: int = 4,
                 H = 40, W = 120,
                 expansions: List[int] = [4, 4, 4],
                 n_blocks: int = 9,
                 decreases: List[int] = [2, 2],
                 bottleneck: int = 64,
                 fc_flag = True
                ) -> None:
        super().__init__()
        
        decrease_steps = len(decreases)
        expansion_steps = len(expansions)
        if decrease_steps > n_blocks:
            raise NotImplementedError("Number of decreases can't be more than number of MyBasicBlocks")
        if expansion_steps > n_blocks:
            raise NotImplementedError("Number of channel expansions can't be more than number of MyBasicBlocks")
        
        layers = []
        
        current_n_channels = in_channels
        expansions_count = 0
        decreases_count = 0
        for layer in range(n_blocks):
            if ((layer+1) % int(n_blocks/len(expansions)) == 1 and expansions_count < len(expansions)) and not ((layer+1) % int(n_blocks/len(decreases)) == 0 and decreases_count < len(decreases)):
                layers.append(MyBasicBlock(inplanes=current_n_channels, expansion=expansions[expansions_count]))
                current_n_channels *= expansions[expansions_count]
                expansions_count += 1
            elif ((layer+1) % int(n_blocks/len(decreases)) == 0 and decreases_count < len(decreases)) and not ((layer+1) % int(n_blocks/len(expansions)) == 1 and expansions_count < len(expansions)):
                layers.append(MyBasicBlock(inplanes=current_n_channels, stride=decreases[decreases_count]))
                H = (H - 1) // decreases[decreases_count] + 1
                W = (W - 1) // decreases[decreases_count] + 1
                decreases_count += 1
            elif ((layer+1) % int(n_blocks/len(expansions)) == 1 and expansions_count < len(expansions)) and ((layer+1) % int(n_blocks/len(decreases)) == 0 and decreases_count < len(decreases)):
                layers.append(MyBasicBlock(inplanes=current_n_channels,
                                           expansion=expansions[expansions_count],
                                           stride=decreases[decreases_count]))
                current_n_channels *= expansions[expansions_count]
                expansions_count += 1
                H = (H - 1) // decreases[decreases_count] + 1
                W = (W - 1) // decreases[decreases_count] + 1
                decreases_count += 1
            else:
                layers.append(MyBasicBlock(inplanes=current_n_channels))
        
        self.layers = nn.Sequential(*layers)
        
        self.fc_flag = fc_flag
        if self.fc_flag:
            current_size = H * W
            fc_features = current_n_channels * current_size
            self.fc_layers = nn.Sequential(
                nn.Linear(fc_features, 1024),
                nn.Linear(1024, bottleneck)
            )
        
    def forward(self, x: Tensor) -> Tensor:
        out = self.layers.forward(x)
        if self.fc_flag:
            out = torch.flatten(out, start_dim=1)
            out = self.fc_layers.forward(out)
        return out
