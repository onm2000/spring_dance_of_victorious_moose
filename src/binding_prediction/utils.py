import torch


def _calc_padding(dilation, kernel_size):
    padding = dilation * (kernel_size - 1) / 2
    int_padding = int(padding)
    assert(padding == int_padding)
    return int_padding


def _unpack_from_convolution(x, batch_size):
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
    x = torch.transpose(x, 1, 2)
    xs = x.shape
    num_N = int(xs[0] / batch_size)
    x = x.reshape((batch_size, num_N) + xs[1:])
    return x


def _pack_for_convolution(x):
    """
    Moves a specified dimension to the end.

    """
    xs = x.shape
    x = x.reshape((xs[0] * xs[1],) + xs[2:])
    x = torch.transpose(x, 1, 2)
    return x
