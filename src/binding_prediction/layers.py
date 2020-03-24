import torch
import torch.nn as nn
from .utils import _calc_padding, _unpack_from_convolution, _pack_for_convolution


class GraphAndConv(nn.Module):
    def __init__(self, input_dim, output_dim, conv_kernel_size, invariant_input_dim=0, intermediate_dim=None):
        super(GraphAndConv, self).__init__()
        if intermediate_dim is None:
            intermediate_dim = output_dim
        if invariant_input_dim >= 0:
            self.invariant_lin = nn.Linear(2*invariant_input_dim, intermediate_dim)
        self.lin = nn.Linear(2*input_dim, intermediate_dim)
        padding = _calc_padding(1, conv_kernel_size)
        self.conv = nn.Conv1d(intermediate_dim, output_dim, conv_kernel_size, padding=padding)

        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

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
