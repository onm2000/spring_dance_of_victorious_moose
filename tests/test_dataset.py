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
        assert(first_element['is_true'])

        if multiple_bond_types:
            assert(len(first_element['adj_mat'].shape) == 3)
        else:
            assert(len(first_element['adj_mat'].shape) == 2)

    @pytest.mark.parametrize('precompute', [True, False])
    def test_sampling(self, precompute):
        def fake_dist():
            return 1, 3
        rand_dset = DrugProteinDataset("data/sample_dataset.txt", 'data/%s.txt',
                                       precompute=True, prob_fake=1., fake_dist=fake_dist)
        ref_dset = DrugProteinDataset("data/sample_dataset.txt", 'data/%s.txt',
                                      precompute=True, prob_fake=0.)
        fake_element = rand_dset[0]
        true_element_1 = ref_dset[1]
        true_element_3 = ref_dset[3]

        assert(torch.norm(fake_element['node_features'] - true_element_1['node_features']) < 1e-6)
        assert(torch.norm(fake_element['adj_mat'] - true_element_1['adj_mat']) < 1e-6)
        assert(torch.norm(fake_element['prot_embedding'] - true_element_3['prot_embedding']) < 1e-6)
        assert(fake_element['is_true'] is False)


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
