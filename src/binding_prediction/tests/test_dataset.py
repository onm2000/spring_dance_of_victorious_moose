import unittest
from binding_prediction.utils import get_data_path
from binding_prediction.dataset import DrugProteinDataset, _load_datafile


class TestDataset(unittest.TestCase):

    def test_load(self):
        fname = get_data_path('example.txt')
        smiles, prots = _load_datafile(fname)
        self.assertEqual(smiles[0], 'DMDVEPIJCJGHPE-UHFFFAOYSA-K')
        exp = ('MVLAWPDRYSSVQLELPEGATVAEAVATSGLALQQAPAAHAVHGLVARPEQ'
               'VLRDGDRVELLRPLLLDPKEARRRRAGPSKKAGHNS')
        self.assertEqual(prots[1], exp)


if __name__ == '__main__':
    unittest.main()
