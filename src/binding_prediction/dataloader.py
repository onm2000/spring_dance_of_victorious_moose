from torch.utils.data import Dataset
from rdkit.Chem import rdmolops
from rdkit import Chem
import torch
import numpy as np
import scipy.sparse as sps


class DrugProteinDataset(Dataset):
    """
    Datase containing drugs, protein IDs, and whether or not they bind.
    """

    def __init__(self, datafile, protein_embedding_template, multiple_bond_types=False, precompute=True):
        """
        Args:
            datafile (string) : Data file that has the uniprot ids, smiles strings,
            and the binding type (FIX?)
        protein_embedding_folder (string) : Template for file containing embeddings.
        """
        data = np.genfromtxt(datafile)
        self.full_data = data
        self.protein_embedding_template = protein_embedding_template
        self.protein_uniprot_ids, prot_invs = np.unique(data[:, 0], return_inverse=True)
        self.drugs, drug_invs = np.unique(data[:, 1], return_inverse=True)
        self.multiple_bond_types

        # Build interaction matrix
        self._build_interaction_matrix()
        self._build_dataset_info()
        self.precompute = precompute
        if precompute:
            self.compute_graphs()
            self.load_embeddings()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prot_id, smiles = self.full_data[idx]
        if self.precompute:
            nodes, edges = self.drug_graphs[smiles]
            embedding = self.prot_embeddings[smiles]
        else:
            nodes, edges, __ = self.build_graph(smiles)
            embedding = self.get_prot_embedding(prot_id)
        adj_mat = self._graph_to_adj_mat(edges)
        if self.multiple_bond_types:
            adj_mat = torch.sum(adj_mat, dim=2)

        sample = {'node_features': nodes, 'adj_mat': adj_mat, 'prot_embedding': embedding}

        if self.transfom:
            sample = self.transform(sample)

        return sample

    def _build_interaction_matrix(self, drug_invs, prot_invs):
        M = np.max(prot_invs) + 1
        N = np.max(drug_invs) + 1
        self.interaction_matrix = sps.csr(np.ones(drug_invs.shape), (prot_invs, drug_invs), shape=(M, N))

    def compute_graphs(self):
        self.drug_graphs = {}
        for drug_smiles in self.drugs:
            nodes, edges, __ = self.build_graph(drug_smiles)
            self.drug_graphs[drug_smiles] = (nodes, edges)

    def load_embeddings(self):
        self.prot_embeddings = {}
        for prot_id in self.protein_uniprot_ids:
            self.prot_embeddings[prot_id] = self.get_prot_embedding(prot_id)

    def get_prot_embedding(self, prot_id):
        return np.load(self.protein_embedding_template % prot_id)

    def _build_drug_graph(self, smiles):
        """
        Builds a molecular graph form a smiles string.  Taken from [FIND SOURCE!]
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [], []
        # Kekulize it
        if self.need_kekulize(mol):
            rdmolops.Kekulize(mol)
            if mol is None:
                return None, None
        # remove stereo information, such as inward and outward edges
        Chem.RemoveStereochemistry(mol)

        edges = []
        nodes = []
        for bond in mol.GetBonds():
            edges.append((bond.GetBeginAtomIdx(), self.bond_dict[str(bond.GetBondType())], bond.GetEndAtomIdx()))
            assert self.bond_dict[str(bond.GetBondType())] != 3
        for atom in mol.GetAtoms():
            nodes.append(onehot(self.dataset_info['atom_types'].index(atom.GetSymbol()), len(self.dataset_info['atom_types'])))

        return nodes, edges, mol

    def _build_dataset_info(self):
        self.dataset_info = {'atom_types': ["H", "C", "N", "O", "F"],
                             'maximum_valence': {0: 1, 1: 4, 2: 3, 3: 2, 4: 1},
                             'number_to_atom': {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"},
                             'bucket_sizes': np.array(list(range(4, 28, 2)) + [29])
                             }

        self.bond_dict = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, "AROMATIC": 3}

    def _graph_to_adj_mat(self, edge_features):
        adj_mat = torch.zeros(self.max_size, self.max_size, self.num_edge_features)
        edge_features = torch.LongTensor(edge_features)
        adj_mat[edge_features[:, 0], edge_features[:, 2], edge_features[:, 1]] = 1.
        return adj_mat

    def need_kekulize(self, mol):
        """
        Check if a molecule needs kekulization.  Taken from [FIND SOURCE!]
        """
        for bond in mol.GetBonds():
            if self.bond_dict[str(bond.GetBondType())] >= 3:
                return True
        return False


def onehot(idx, len):
    z = [0 for _ in range(len)]
    z[idx] = 1
    return z
