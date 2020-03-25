import torch
import torch.nn as nn
from .utils import _calc_padding, _unpack_from_convolution, _pack_for_convolution


class GraphAndConv(nn.Module):
    def __init__(self, input_dim, output_dim, conv_kernel_size, intermediate_dim=None):
        super(GraphAndConv, self).__init__()
        if intermediate_dim is None:
            intermediate_dim = output_dim
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


class RankingLayer(nn.Module):
    def __init__(self, input_size, emb_dim):
        """ Initialize model parameters for Siamese network.
        Parameters
        ----------
        input_size: int
            Input dimension size
        emb_dim: int
            Embedding dimension for both datasets
        Note
        ----
        This implicitly assumes that the embedding dimension for
        both datasets are the same.
        """
        # See here: https://adoni.github.io/2017/11/08/word2vec-pytorch/
        super(RankingLayer, self).__init__()
        self.input_size = input_size
        self.emb_dimension = emb_dimension
        self.output = nn.Linear(input_size, emb_dim)
        self.init_emb()

    def init_emb(self):
        initstd = 1 / math.sqrt(self.emb_dimension)
        self.output.weight.data.normal_(0, initstd)

    def forward(self, pos, neg):
        """
        Parameters
        ----------
        pos : torch.Tensor
           Positive shared representation vector
        neg : torch.Tensor
           Negative shared representation vector(s).
           There can be multiple negative examples (~5 according to NCE).
        """
        losses = 0
        pos_out = self.output(pos)
        neg_out = self.output(neg)
        diff = pos - neg_out
        score = F.logsigmoid(diff)
        losses = sum(score)
        return -1 * losses

