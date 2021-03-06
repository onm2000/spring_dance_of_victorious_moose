import torch
import pytest
from binding_prediction.models import GraphAndConvStack, BindingModel
from binding_prediction import pretrained_language_models


def _permute_tensors(features, adj_mats, p_indices):
    permuted_adj_mats = []
    permuted_feature_mats = []
    for i, pidx in enumerate(p_indices):
        permuted_adj_mats.append(adj_mats[i][pidx][:, pidx])
        permuted_feature_mats.append(features[i][pidx])
    permuted_adj_mats = torch.stack(permuted_adj_mats, dim=0)
    permuted_feature_mats = torch.stack(permuted_feature_mats, dim=0)
    return permuted_feature_mats, permuted_adj_mats


class TestGraphAndConvStack(object):
    @pytest.mark.parametrize('num_intermediate', [None, 4])
    @pytest.mark.parametrize('hidden_channel_list', [[3], [2, 2], []])
    @pytest.mark.parametrize('explicit_conv_kernel_sizes', [True, False])
    def test_permutation_equivariance(self, sample_batch, num_intermediate,
                                      hidden_channel_list, explicit_conv_kernel_sizes):
        features, adj_mats = sample_batch
        B, N, __ = adj_mats.shape
        p_indices = [torch.randperm(N) for i in range(B)]

        feature_perm, adj_mats_perm = _permute_tensors(features, adj_mats, p_indices)

        if explicit_conv_kernel_sizes:
            conv_kernel_sizes = list(range(1, 2 * (len(hidden_channel_list) + 2), 2))
        else:
            conv_kernel_sizes = None
        gconv = GraphAndConvStack(3, hidden_channel_list, 1,
                                  conv_kernel_sizes=conv_kernel_sizes)
        output = gconv(adj_mats, features)[-1]
        permed_output = _permute_tensors(output, adj_mats, p_indices)[0]
        output_from_perm = gconv(adj_mats_perm, feature_perm)[-1]
        assert(torch.norm(permed_output - output_from_perm) < 1e-4)

    @pytest.mark.parametrize('hidden_channel_list', [[3], [2, 2], []])
    def test_translational_equivariance(self, sample_batch, hidden_channel_list):
        features, adj_mats = sample_batch
        features[:, :, -1] = 0.
        trans_features = torch.zeros(features.shape)
        trans_features[:, :, 1:] = features[:, :, :-1]

        gconv = GraphAndConvStack(3, hidden_channel_list, 1)
        output = gconv(adj_mats, features)[-1]
        output_from_translation = gconv(adj_mats, trans_features)[-1]
        assert(torch.norm(output[:, :, 4:-5] - output_from_translation[:, :, 5:-4]) < 1e-4)


class TestBindingModel(object):
    def test_permutation_invariance(self, sample_sequences):
        node_features = torch.randn(3, 13, 4)

        adj_mats = torch.randint(2, size=(3, 13, 13)).float()

        B, N, __ = adj_mats.shape
        p_indices = [torch.randperm(N) for i in range(B)]

        feature_perm, adj_mats_perm = _permute_tensors(node_features, adj_mats, p_indices)

        model = BindingModel(4, 512, 3, 3, [3], 3)
        cls, path = pretrained_language_models['elmo']
        model.load_language_model(cls, path, device='cpu')
        output = model(adj_mats, node_features, sample_sequences)
        output_from_perm = model(adj_mats_perm, feature_perm, sample_sequences)
        assert(torch.norm(output - output_from_perm) < 1e-3)
