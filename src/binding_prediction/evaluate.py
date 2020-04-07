import numpy as np
import pandas as pd
import torch
from poplar.util import encode, tokenize


# Global evaluation metrics
# these are used mainly for testing evaluation
def mrr(model, dataloader):
    """ Mean reciprocial ranking.

    Parameters
    ----------
    model : popular.model
       Model to be evaluated
    dataloader : torch.DataLoader
       Pytorch dataloader.

    Returns
    -------
    float : mean reciprocial ranking
    """
    pass

def roc_auc(model, dataloader, k=10):
    """ ROC AUC

    Parameters
    ----------
    model : popular.model
       Model to be evaluated
    dataloader : torch.DataLoader
       Pytorch dataloader for validation data

    Returns
    -------
    float : average AUC

    TODO
    ----
    Make sure that the test/validation dataset are
    sorted by (1) taxonomy then by (2) protein1.
    """
    pass

def pairwise_auc(binding_model,
                 dataloader, name, it, writer,
                 device='cpu'):
    """ Pairwise AUC comparison

    Parameters
    ----------
    binding_model : binding model
       Binding prediction model.
    dataloader : dataloader
       Dataset iterator for test/validation ppi dataset.
    name : str
       Name of the database used in dataloader.
    it : int
       Iteration number.
    writer : SummaryWriter
       Tensorboard writer.
    device : str
       Device name to transfer model data to.

    Returns
    -------
    float : average AUC
    """
    with torch.no_grad():
        rank_counts = 0
        for j, (gene, pos, rnd, tax, protid) in enumerate(dataloader):
            gv = binding_model.encode([gene])
            pv = binding_model.encode([pos])
            nv = binding_model.encode([rnd])
            pred_pos = binding_model.predict(gv, pv)
            pred_neg = binding_model.predict(gv, nv)
            score = torch.sum(pred_pos > pred_neg).item()
            rank_counts += score

        tpr = rank_counts / j
        print(f'rank_counts {rank_counts}, tpr {tpr}, iteration {it}')
        writer.add_scalar(f'{name}/pairwise/TPR', tpr, it)

    return tpr
