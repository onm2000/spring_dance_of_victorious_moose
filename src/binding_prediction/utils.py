def _calc_padding(dilation, kernel_size):
    padding = dilation * (kernel_size - 1) / 2
    int_padding = int(padding)
    assert(padding == int_padding)
    return int_padding


def _move_from_end(x, dim):
    """
    Moves the last dimension to the desired position while preserving ordering of the others.

    Parameters
    ----------
    x : :class:`torch.Tensor`
        Input tensor of shape ... x n
    dim : int
        Dimension to move the last location.

    Returns
    -------
    x : :class:`torch.Tensor`
        Permuted tensor
    """
    N = len(x.shape)
    if dim < 0:
        dim = N + dim
    permute_indices = list(range(N-1))
    permute_indices.insert(dim, N-1)
    return x.permute(permute_indices)


def _move_to_end(x, dim):
    """
    Moves a specified dimension to the end.

    """
    N = len(x.shape)
    if dim < 0:
        dim = N + dim
    permute_indices = list(range(N))
    permute_indices.remove(dim)
    permute_indices.append(dim)
    return x.permute(permute_indices)
