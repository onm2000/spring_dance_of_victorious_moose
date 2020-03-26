import dgl
from torch.utils.data import Dataset
from rdkit.Chem import rdmolops
from rdkit import Chem
import math
import torch
import numpy as np
import scipy.sparse as sps
import pandas as pd


def _load_datafile(datafile):
    """
    Parameters
    ----------
    datafile : str
       File path.  Assumes that the file is only two columns
          1. Drug INCHI string
          2. Protein sequence
    """
    df = pd.read_table(datafile, header=None, delim_whitespace=True)
    drug_inchi = np.array(df[0])  # first column is inchi key
    protein_seqs = np.array(df[1])  # second column is sequence
    return drug_inchi, protein_seqs


class DrugProteinDataset(Dataset):
    """
    Database containing drugs, protein IDs, and whether or not they bind.
    """

    def __init__(self, datafile, multiple_bond_types=False,
                 precompute=True, transform=None, prob_fake=0.0, fake_dist=None):
        """
        Args:
            datafile (string) : Data file that has the protein sequences, smiles strings,
            and the binding type (FIX?)
        protein_embedding_folder (string) : Template for file containing embeddings.
        """
        super(DrugProteinDataset, self).__init__()
        self.all_drugs, self.all_prots = _load_datafile(datafile)
        self.unique_prots, prot_invs = np.unique(self.all_prots, return_inverse=True)
        self.unique_drugs, drug_invs = np.unique(self.all_drugs, return_inverse=True)
        self.multiple_bond_types = multiple_bond_types
        self.num_edge_features = 3
        self.prob_fake = prob_fake

        if fake_dist is None:
            def fake_dist():
                drug_idx = np.random.choice(self.all_drugs)
                embed_idx = np.random.choice(self.all_prots)
                return drug_idx, embed_idx
        self.fake_dist = fake_dist

        # Build interaction matrix
        # self._build_interaction_matrix(drug_invs, prot_invs)
        self._build_dataset_info()
        self.precompute = precompute
        if precompute:
            self.compute_graphs()

        self.transform = transform

    def __len__(self):
        return len(self.all_drugs)

    def _preprocess_molecule(self, smiles):
        if self.precompute:
            nodes, edges = self.drug_graphs[smiles]
        else:
            nodes, edges, __ = self._build_drug_graph(smiles)
        adj = self._graph_to_adj_mat(edges)
        if not self.multiple_bond_types:
            adj = torch.sum(adj, dim=2)
        return nodes, adj

    def __getitem__(self, idx):
        is_true = (np.random.rand() < (1. - self.prob_fake))
        if is_true:
            drug_idx = prot_idx = idx
        else:
            drug_idx, prot_idx = self.fake_dist()

        smiles = self.all_drugs[drug_idx]
        prot = self.all_prots[prot_idx]

        nodes, adj_mat = self._preprocess_molecule(smiles)

        sample = {'node_features': nodes, 'adj_mat': adj_mat,
                  'protein': prot, 'is_true': int(is_true)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _build_interaction_matrix(self, drug_invs, prot_invs):
        M = np.max(prot_invs) + 1
        N = np.max(drug_invs) + 1
        self.interaction_matrix = sps.csr(np.ones(drug_invs.shape),
                                          (prot_invs, drug_invs), shape=(M, N))

    def compute_graphs(self):
        self.drug_graphs = {}
        for drug_smiles in self.unique_drugs:
            nodes, edges, __ = self._build_drug_graph(drug_smiles)
            self.drug_graphs[drug_smiles] = (nodes, edges)

    def _build_drug_graph(self, smiles):
        """
        Builds a molecular graph form a smiles string.  Taken from [FIND SOURCE!]
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError('Molecule construction failed on Inchi %s' % smiles)
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
            edges.append((bond.GetBeginAtomIdx(), self.bond_dict[str(bond.GetBondType())],
                          bond.GetEndAtomIdx()))
            assert self.bond_dict[str(bond.GetBondType())] != 3
        for atom in mol.GetAtoms():
            nodes.append(onehot(self.dataset_info['atom_types'].index(atom.GetSymbol()),
                                len(self.dataset_info['atom_types'])))

        nodes = torch.tensor(nodes).float()
        edges = torch.tensor(edges)

        return nodes, edges, mol

    def _build_dataset_info(self):
        self.dataset_info = {'atom_types': ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "Fe"]
                             }

        self.bond_dict = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, "AROMATIC": 3}

    def _graph_to_adj_mat(self, edge_features):
        max_size = max(torch.max(edge_features[:, 0]), torch.max(edge_features[:, 2])) + 1
        adj_mat = torch.zeros(max_size, max_size, self.num_edge_features)
        edge_features = torch.LongTensor(edge_features)
        adj_mat[edge_features[:, 0], edge_features[:, 2], edge_features[:, 1]] = 1.
        adj_mat = adj_mat.float()
        return adj_mat

    def need_kekulize(self, mol):
        """
        Check if a molecule needs kekulization.  Taken from [FIND SOURCE!]
        """
        for bond in mol.GetBonds():
            if self.bond_dict[str(bond.GetBondType())] >= 3:
                return True
        return False


class PosDrugProteinDataset(DrugProteinDataset):
    """ Performs positive only sampling strategy.

    Notes
    -----
    Bayesian Personalized Ranking:
       https://arxiv.org/pdf/1205.2618.pdf
    """
    def __init__(self, num_neg=1, **kwargs):
        super(PosDrugProteinDataset, self).__init__(**kwargs)
        self.num_neg = num_neg

    def __getitem__(self, idx):

        drug_pos = prot_idx = idx
        smiles_neg = np.random.choice(self.all_drugs)

        smiles_pos = self.all_drugs[drug_pos]
        prot = self.all_prots[prot_idx]

        pos_nodes, pos_adj = self._preprocess_molecule(smiles_pos)
        neg_nodes, neg_adj = self._preprocess_molecule(smiles_neg)

        sample = {'pos_node_features': pos_nodes, 'pos_adj_mat': pos_adj,
                  'neg_node_features': neg_nodes, 'neg_adj_mat': neg_adj,
                  'protein': prot}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __iter__(self):
        """ Need this for multi-GPU support"""
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = len(self.pairs)

        if worker_info is None:  # single-process data loading
            for i in range(end):
                for _ in range(self.num_neg):
                    yield self.__getitem__(i)
        else:
            worker_id = worker_info.id
            w = float(worker_info.num_workers)
            t = (end - start)
            w = float(worker_info.num_workers)
            per_worker = int(math.ceil(t / w))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)
            for i in range(iter_start, iter_end):
                for _ in range(self.num_neg):
                    yield self.__getitem__(i)


class MergeSnE1(object):
    def __init__(self):
        super(MergeSnE1, self).__init__()

    def __call__(self, sample):
        embedding = sample['protein']
        nodes = sample['node_features']
        N_resid = embedding.shape[0]
        N_nodes = nodes.shape[0]
        nodes_expanded = torch.stack([nodes] * N_resid, dim=1)
        embed_expanded = torch.stack([embedding] * N_nodes, dim=0)
        full_features = torch.cat((nodes_expanded, embed_expanded), dim=2)

        sample['features'] = full_features
        return sample

class DGLGraphBuilder(MergeSnE1):
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        full_features = super().__call__(sample)['features']
        adj_mat = sample['adj_mat']
        g = build_dgl_graph(full_features, adj_mat)
        return g

def onehot(idx, len):
    idx = np.array(idx)  # make sure this is an array
    z = np.array([0 for _ in range(len)])
    z[idx] = 1
    return z


def collate_fn(batch):
    collated_batch = {}
    for prop in batch[0].keys():
        if prop == 'adj_mat':
            edge_mat = True
        else:
            edge_mat = False
        collated_batch[prop] = _batch_stack([mol[prop] for mol in batch], edge_mat=edge_mat)
    return collated_batch


def _batch_stack(props, edge_mat=False):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.  Adapted from the cormorant library, and
    initially written by Brandon Anderson.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack
    edge_mat : bool
        The included tensor refers to edge properties, and therefore needs
        to be stacked/padded along two axes instead of just one.

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        props = [torch.tensor(pi) for pi in props]
    if props[0].dim() == 0:
        return torch.stack(props)
    elif not edge_mat:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)
    else:
        max_atoms = max([len(p) for p in props])
        max_shape = (len(props), max_atoms, max_atoms) + props[0].shape[2:]
        padded_tensor = torch.zeros(max_shape, dtype=props[0].dtype, device=props[0].device)

        for idx, prop in enumerate(props):
            this_atoms = len(prop)
            padded_tensor[idx, :this_atoms, :this_atoms] = prop

        return padded_tensor


def build_dgl_graph(features, adj_mat):
    g = dgl.DGLGraph()
    N_nodes = features.shape[0]
    g.add_nodes(N_nodes, {'features': features})
    nonzero_coords = torch.nonzero(adj_mat)
    u, v = nonzero_coords[:, 1], nonzero_coords[:, 0]
    g.add_edges(u, v)
    return g


def build_dgl_graph_batch(batch_features, batch_adj_mat):
    graphs = []
    for i in range(len(batch_adj_mat)):
        graphs.append(build_dgl_graph(batch_features[i], batch_adj_mat[i]))
    graphs_batch = dgl.batch(graphs)
    return graphs_batch
