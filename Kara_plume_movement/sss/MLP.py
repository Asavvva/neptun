from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from mish import Mish

class MultiLayerPerceptron(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_layers: List[int],
                 activation: Type[torch.nn.Module] = Mish):
        
        super(MultiLayerPerceptron, self).__init__()
        
        layers = []
        previous_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(activation())
            previous_size = hidden_size
        
        layers.append(nn.Linear(previous_size, output_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model.forward(x)
