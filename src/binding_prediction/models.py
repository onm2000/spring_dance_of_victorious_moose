import torch
import torch.nn as nn
from .layers import GraphAndConv, MergeSnE1
from dgl.nn.pytorch.conv import GraphConv
from binding_prediction.dataset import build_dgl_graph_batch


class BindingModel(torch.nn.Module):
    """
    Model for predicting weather or not we achieve binding.

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
                 conv_kernel_sizes=None, nonlinearity=None, layer_cls=GraphAndConv):
        super(BindingModel, self).__init__()
        self.gcs_stack = GraphAndConvStack(in_channels, hidden_channel_list,
                                           out_channels, conv_kernel_sizes,
                                           nonlinearity, layer_cls)
        self.merge_graph_w_sequences = MergeSnE1()
        total_number_inputs = sum(hidden_channel_list) + out_channels
        self.final_mix = nn.Linear(total_number_inputs, 1, bias=False)
        self.lm = None

    def forward(self, adj, x, prot_sequences):
        if self.lm is None:
            raise ValueError('Language model is not initialized!')
        prot_embeddings = [self.lm(p_i) for p_i in prot_sequences]
        y = self.merge_graph_w_sequences(x, prot_embeddings)
        x_all = self.gcs_stack(adj, y)
        x_all = torch.cat(x_all, dim=-1)
        x_out = self.final_mix(x_all)
        x_out = torch.sum(torch.sum(x_out, dim=2), dim=1)
        return x_out

    def load_language_model(self, cls, path):
        """
        Parameters
        ----------
        cls : Module name
            Name of the Language model.
            (i.e. binding_prediction.language_model.Elmo)
        path : filepath
            Filepath of the pretrained model.
        """
        self.lm = cls(path)


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
                 conv_kernel_sizes=None, nonlinearity=None, layer_cls=GraphAndConv):
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
            conv_i = layer_cls(in_channel_i, out_channel_i,
                                  conv_kernel_size=conv_kernel_sizes[i])
            self.conv_layers.append(conv_i)
            in_channel_i = out_channel_i
        conv_i = layer_cls(in_channel_i, out_channels,
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


class DecomposableAttentionModel(nn.Module):
    def __init__(self, node_dim, num_gnn_layers, lm):
        super().__init__()
        self.lm = lm
        self.residue_dim = self.lm.model.output_shape[-1]
        self.gnn = GraphConv(node_dim, node_dim, activation='ReLU')
        self.num_gnn_layers = num_gnn_layers
        self.attn_weight_layer = nn.Sequential(nn.Linear(node_dim, 1), nn.Softmax(dim=-1))
        self.interaction_layer = nn.Sequential(
            nn.Linear(node_dim+self.residue_dim, node_dim + int(self.residue_dim/2)),
            nn.ReLU(),
            nn.Linear(node_dim + int(self.residue_dim/2), node_dim)
        )
        self.output_layer = nn.Linear(node_dim, 1)

        self.merge_graph_w_sequences = MergeSnE1()


    def forward(self, adj_mats, nodes, protein_sequences):
        if self.lm is None:
            raise ValueError('Language model is not initialized!')
        protein_sequences = [self.lm(p_i) for p_i in protein_sequences]
        batch_size, max_nodes, node_dim = nodes.shape
        seq_length, residue_dim = protein_sequences[0].shape[1:]
        batch_graph = build_dgl_graph_batch(nodes, adj_mats)
        for _ in range(self.num_gnn_layers):
            nodes = self.gnn(batch_graph)
            batch_graph.ndata['h'] = nodes
        nodes = nodes.reshape(batch_size, max_nodes, -1)
        node_residue_cat = self.merge_graph_w_sequences(nodes, protein_sequences)
        node_residue_cat = node_residue_cat.reshape(batch_size, max_nodes * seq_length, node_dim + residue_dim)
        attn_weights = self.attn_weight_layer(node_residue_cat)
        node_residue_interactions = self.interaction_layer(node_residue_cat)
        weighted_sum = (attn_weights * node_residue_interactions).sum(dim=1)
        score = self.output_layer(weighted_sum)
        return score
