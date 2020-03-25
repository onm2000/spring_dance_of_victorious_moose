import torch.nn as nn
from .layers import GraphAndConv


class GraphAndConvStack(nn.Module):
    """
    Stack of graph and E(1) convolutional layers.

    Parameters
    ----------
    in_channels : int
        The size of the channel index of the input tensor
    hidden_channel_list : iterable of ints
        Number of channels in every hidden layer of the encoder
        Length corresponds to the number of hidden layers.
    out_channels : int
        number of output channels.
    conv_kernel_sizes : iterable of ints, optional
        Number of channels in every hidden layer of the encoder
        Length corresponds to the number of layers.
        If not provided, defaults to 1 for every layer
    """

    def __init__(self, in_channels, hidden_channel_list, out_channels,
                 conv_kernel_sizes=None, nonlinearity=None):
        super(GraphAndConvStack, self).__init__()
        if nonlinearity is None:
            nonlinearity = nn.ReLU
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [1] * (len(hidden_channel_list) + 1)
        self.nonlinearity = nonlinearity()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channel_list = hidden_channel_list

        self.conv_layers = nn.ModuleList()

        in_channel_i = in_channels
        for i, out_channel_i in enumerate(hidden_channel_list):
            conv_i = GraphAndConv(in_channel_i, out_channel_i,
                                  conv_kernel_size=conv_kernel_sizes[i])
            self.conv_layers.append(conv_i)
            in_channel_i = out_channel_i
        conv_i = GraphAndConv(in_channel_i, out_channels,
                              conv_kernel_size=conv_kernel_sizes[-1])
        self.conv_layers.append(conv_i)

    def forward(self, adj, x):
        """
        Runs the net.

        Parameters
        ----------
        x : pytorch tensor
            Input to the convolution, of shape B x N x L x C_in
            Here B is the batch size, C_in is the number of channels,
            N is the size of the largest graph, and L is the size of the E(1)
            equivariant string.

        Returns
        -------
        x_all : list of pytorch tensor
            The output of the net at every layer of the net.
        """
        x = self.conv_layers[0](adj, x)
        x_all = [x]

        for module in self.conv_layers[1:]:
            x = self.nonlinearity(x)
            x = module(adj, x)
            x_all.append(x)

        return x_all
