import torch
import pytest
from binding_prediction.dataset import DrugProteinDataset, MergeSnE1


class TestDrugProteinDataset(object):
    @pytest.mark.parametrize('precompute', [True, False])
    @pytest.mark.parametrize('multiple_bond_types', [True, False])
    def test_output_shapes(self, precompute, multiple_bond_types):
        dset = DrugProteinDataset("data/sample_dataset.txt", 'data/%s.txt',
                                  precompute=precompute, multiple_bond_types=multiple_bond_types)
        first_element = dset[0]
        assert(first_element['prot_embedding'].shape == (10, 4))
        assert(first_element['node_features'].shape[0] == 155)
        assert(first_element['adj_mat'].shape[0] == 155)
        assert(first_element['adj_mat'].shape[1] == 155)

        if multiple_bond_types:
            assert(len(first_element['adj_mat'].shape) == 3)
        else:
            assert(len(first_element['adj_mat'].shape) == 2)


class TestTransform(object):
    def test_merge(self):
        node_features = torch.randn(13, 4)
        prot_embedding = torch.randn(40, 2)
        adj_mat = torch.randint(2, (13, 13)).float()
        input_sample = {'node_features': node_features,
                        'prot_embedding': prot_embedding,
                        'adj_mat': adj_mat}

        tf = MergeSnE1()
        output_sample = tf(input_sample)
        assert(torch.norm(output_sample['adj_mat'] - adj_mat) < 1E-6)

        expected_shape = (13, 40, 6)
        assert(output_sample['features'].shape == expected_shape)
        assert(torch.norm(output_sample['features'][:, 2, :4] - node_features) < 1.e-6)
        assert(torch.norm(output_sample['features'][3, :, 4:] - prot_embedding) < 1.e-6)
