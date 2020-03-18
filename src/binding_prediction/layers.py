import torch
import torch.nn as nn
from .utils import _calc_padding, _move_from_end, _move_to_end


class GraphAndConv(nn.Module):
    def __init__(self, input_dim, output_dim, conv_kernel_size, intermediate_dim=None, conv_dim=-3):
        super(GraphAndConv, self).__init__()
        if intermediate_dim is None:
            intermediate_dim = output_dim
        self.lin = nn.Linear(2*input_dim, intermediate_dim)
        padding = _calc_padding(1, conv_kernel_size)
        self.conv = nn.Conv1D(intermediate_dim, output_dim, conv_kernel_size, padding=padding)

        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.conv_dim = conv_dim

    def forward(self, adj, inputs):
        x = torch.matmul(adj, inputs)
        x = torch.cat((x, inputs), dim=-1)
        x = self.lin(x)

        x = _move_to_end(x, dim=self.conv_dim)
        x = self.conv(x)
        x = _move_from_end(x, dim=self.conv_dim)
        return x
