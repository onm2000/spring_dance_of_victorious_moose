import numpy as np
import torch
import pytest


@pytest.fixture(scope='module')
def sample_batch():
    np.random.seed(8675309)

    features = []
    adj_mats = []
    for batch in range(3):
        adj_mat = np.random.randint(2, size=(10, 10))
        adj_mat = (adj_mat + adj_mat.T).astype('bool').astype('float')
        adj_mat[np.arange(10), np.arange(10)] = 0.

        feature_mat = np.random.randn(10, 8, 3)
        if batch > 0:
            size = np.random.randint(3, 9)
            adj_mat[size:] = 0.
            adj_mat[:, size:] = 0.
            feature_mat[:, size:] = 0.
        adj_mats.append(adj_mat)
        features.append(feature_mat)
    features = torch.from_numpy(np.stack(features, axis=0)).float()
    adj_mats = torch.from_numpy(np.stack(adj_mats, axis=0)).float()
    return features, adj_mats
