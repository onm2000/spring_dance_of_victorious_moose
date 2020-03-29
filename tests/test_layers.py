import torch
import pytest
from binding_prediction.layers import GraphAndConv, GraphAndConvDGL, MergeSnE1


def _permute_tensors(features, adj_mats, p_indices):
    permuted_adj_mats = []
    permuted_feature_mats = []
    for i, pidx in enumerate(p_indices):
        permuted_adj_mats.append(adj_mats[i][pidx][:, pidx])
        permuted_feature_mats.append(features[i][pidx])
    permuted_adj_mats = torch.stack(permuted_adj_mats, dim=0)
    permuted_feature_mats = torch.stack(permuted_feature_mats, dim=0)
    return permuted_feature_mats, permuted_adj_mats


class TestGraphAndConv(object):
    @pytest.mark.parametrize('num_intermediate', [None, 4])
    @pytest.mark.parametrize('layer_cls', [GraphAndConv, GraphAndConvDGL])
    def test_permutation_equivariance(self, sample_batch, num_intermediate, layer_cls):
        features, adj_mats = sample_batch
        B, N, __ = adj_mats.shape
        p_indices = [torch.randperm(N) for i in range(B)]

        feature_perm, adj_mats_perm = _permute_tensors(features, adj_mats, p_indices)

        gconv = layer_cls(3, 4, 1, intermediate_channels=num_intermediate)
        output = gconv(adj_mats, features)
        permed_output = _permute_tensors(output, adj_mats, p_indices)[0]
        output_from_perm = gconv(adj_mats_perm, feature_perm)
        assert(torch.norm(permed_output - output_from_perm) < 1e-4)

    @pytest.mark.parametrize('layer_cls', [GraphAndConv, GraphAndConvDGL])
    def test_translational_equivariance(self, sample_batch, layer_cls):
        features, adj_mats = sample_batch
        features[:, :, -1] = 0.
        trans_features = torch.zeros(features.shape)
        trans_features[:, :, 1:] = features[:, :, :-1]

        gconv = GraphAndConv(3, 4, 1)
        output = gconv(adj_mats, features)
        output_from_translation = gconv(adj_mats, trans_features)
        assert(torch.norm(output[:, :, :-1] - output_from_translation[:, :, 1:]) < 1e-4)


class TestMergeSnE1(object):
    def test_merge(self):
        node_features = torch.randn(3, 13, 4)
        protein_sequences = torch.randn(3, 40, 2)
        tf = MergeSnE1()
        output_sample = tf(node_features, protein_sequences)

        expected_shape = (3, 13, 40, 6)
        assert(output_sample.shape == expected_shape)
        assert(torch.norm(output_sample[:, :, 2, :4] - node_features) < 1.e-6)
        assert(torch.norm(output_sample[:, 3, :, 4:] - protein_sequences) < 1.e-6)
        # for i in range(3):
        #     len_i = len(protein_sequences[i])
        #     assert(torch.norm(output_sample[i, 3, :len_i, 4:] - protein_sequences[i]) < 1.e-6)
        #     assert(torch.norm(output_sample[i, 3, len_i:, 4:]) < 1e-6)
