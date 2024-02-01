import torch
import torch.nn as nn
from .GraphConvolution import GraphConvolution

class GraphConvolutionalEncoder(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int, out_channels: int, activation, base_model=GraphConvolution):
        super(GraphConvolutionalEncoder, self).__init__()
        self.base_model = base_model

        self.gcn1 = base_model(in_channels, hid_channels, seed, bias = True)
        self.gcn2 = base_model(hid_channels, out_channels, seed, bias = True)
        self.activation = activation

    def forward(self, x: torch.Tensor, edges: torch.Tensor):
        if x.data.dim() == 3:
            x = x.squeeze()

        x1 = self.activation(self.gcn1(x, edges))
        x2 = self.activation(self.gcn2(x1, edges))
        return x2

