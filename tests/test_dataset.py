import torch
import pytest
from binding_prediction.dataset import DrugProteinDataset


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


        
