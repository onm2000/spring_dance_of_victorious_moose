import torch
import torch.nn as nn
from .utils import _calc_padding, _unpack_from_convolution, _pack_for_convolution


class GraphAndConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=1, intermediate_channels=None):
        super(GraphAndConv, self).__init__()
        if intermediate_channels is None:
            intermediate_channels = out_channels
        self.lin = nn.Linear(2*in_channels, intermediate_channels)
        padding = _calc_padding(1, conv_kernel_size)
        self.conv = nn.Conv1d(intermediate_channels, out_channels, conv_kernel_size, padding=padding)

        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.out_channels = out_channels

    def forward(self, adj, inputs):
        batch_size = inputs.shape[0]
        mask = adj.sum(dim=2).bool()
        x = torch.einsum('bilc,bij->bjlc', inputs, adj)
        x = torch.cat((x, inputs), dim=-1)
        x = self.lin(x)
        x = _pack_for_convolution(x)
        x = self.conv(x)
        x = _unpack_from_convolution(x, batch_size)
        x[~mask] = 0.
        return x
