import torch
import pytest
from binding_prediction.dataset import DrugProteinDataset, MergeSnE1, DGLGraphBuilder, collate_fn, _load_datafile
from binding_prediction.utils import get_data_path


class TestDataUtils(object):
    def test_load(self):
        fname = get_data_path('example.txt')
        smiles, prots = _load_datafile(fname)
        assert(smiles[0] == 'CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C')
        exp = ('MVLAWPDRYSSVQLELPEGATVAEAVATSGLALQQAPAAHAVHGLVARPEQ'
               'VLRDGDRVELLRPLLLDPKEARRRRAGPSKKAGHNS')
        assert(prots[1] == exp)


class TestDrugProteinDataset(object):
    @pytest.mark.parametrize('precompute', [True, False])
    @pytest.mark.parametrize('multiple_bond_types', [True, False])
    def test_output_shapes(self, precompute, multiple_bond_types):
        dset = DrugProteinDataset("data/example.txt", precompute=precompute,
                multiple_bond_types=multiple_bond_types)
        first_element = dset[0]
        assert(first_element['node_features'].shape[0] == 14)
        assert(first_element['adj_mat'].shape[0] == 14)
        assert(first_element['adj_mat'].shape[1] == 14)
        assert(first_element['is_true'])

        if multiple_bond_types:
            assert(len(first_element['adj_mat'].shape) == 3)
        else:
            assert(len(first_element['adj_mat'].shape) == 2)

    @pytest.mark.parametrize('precompute', [True, False])
    def test_sampling(self, precompute):
        def fake_dist():
            return 1, 2
        rand_dset = DrugProteinDataset("data/example.txt", precompute=True,
                                       prob_fake=1., fake_dist=fake_dist)
        ref_dset = DrugProteinDataset("data/example.txt", precompute=True,
                                      prob_fake=0.)
        fake_element = rand_dset[0]
        true_element_1 = ref_dset[1]
        true_element_3 = ref_dset[2]

        assert(torch.norm(fake_element['node_features'] - true_element_1['node_features']) < 1e-6)
        assert(torch.norm(fake_element['adj_mat'] - true_element_1['adj_mat']) < 1e-6)
        assert(fake_element['protein'] == true_element_3['protein'])
        assert(fake_element['is_true'] == 0)


class TestTransform(object):
    def test_merge(self):
        node_features = torch.randn(13, 4)
        protein = torch.randn(40, 2)
        adj_mat = torch.randint(2, (13, 13)).float()
        input_sample = {'node_features': node_features,
                        'protein': protein,
                        'adj_mat': adj_mat}

        tf = MergeSnE1()
        output_sample = tf(input_sample)
        assert(torch.norm(output_sample['adj_mat'] - adj_mat) < 1E-6)

        expected_shape = (13, 40, 6)
        assert(output_sample['features'].shape == expected_shape)
        assert(torch.norm(output_sample['features'][:, 2, :4] - node_features) < 1.e-6)
        assert(torch.norm(output_sample['features'][3, :, 4:] - protein) < 1.e-6)

    def test_dgl(self):
        node_features = torch.randn(13, 4)
        prot_embedding = torch.randn(40, 2)
        adj_mat = torch.randint(2, (13, 13)).float()
        input_sample = {'node_features': node_features,
                        'prot_embedding': prot_embedding,
                        'adj_mat': adj_mat}

        tf = DGLGraphBuilder()
        output_graph = tf(input_sample)
        assert(torch.norm(output_graph.adjacency_matrix().to_dense() - adj_mat) < 1E-6)

        expected_shape = (13, 40, 6)
        assert(output_graph.ndata['features'].shape == expected_shape)
        assert(torch.norm(output_graph.ndata['features'][:, 2, :4] - node_features) < 1.e-6)
        assert(torch.norm(output_graph.ndata['features'][3, :, 4:] - prot_embedding) < 1.e-6)

def test_collate_fxn():
    node_features = [torch.randn(13, 4), torch.randn(8, 4), torch.randn(15, 4)]
    protein = [torch.randn(40, 2), torch.randn(15, 2), torch.randn(30, 2)]
    adj_mat = [torch.randint(2, (13, 13)).float(), torch.randint(2, (8, 8)).float(), torch.randint(2, (15, 15)).float()]
    is_true = [1, 1, 0]
    input_batches = []
    for n_i, p_i, a_i, t_i in zip(node_features, protein, adj_mat, is_true):
        sample_i = {'node_features': n_i, 'protein': p_i, 'adj_mat': a_i, 'is_true': t_i}
        input_batches.append(sample_i)

    collated_batch = collate_fn(input_batches)
    assert(collated_batch['node_features'].shape == (3, 15, 4))
    assert(collated_batch['protein'].shape == (3, 40, 2))
    assert(torch.norm(collated_batch['is_true'] - torch.tensor([1., 1., 0.])) < 1e-6)

    for i, (n_i, p_i, a_i) in enumerate(zip(node_features, protein, adj_mat)):
        drug_size, drug_channels = n_i.shape
        prot_size, prot_channels = p_i.shape
        batched_n_i = collated_batch['node_features'][i][:drug_size, :drug_channels]
        assert(torch.norm(batched_n_i - n_i) < 1e-4)
        batched_p_i = collated_batch['protein'][i][:prot_size, :prot_channels]
        assert(torch.norm(batched_p_i - p_i) < 1e-4)
        batched_a_i = collated_batch['adj_mat'][i][:drug_size, :drug_size]
        assert(torch.norm(batched_a_i - a_i) < 1e-4)





