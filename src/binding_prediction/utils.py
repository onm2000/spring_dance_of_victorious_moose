import torch
import inspect
import os


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


def get_data_path(fn, subfolder='data'):
    """Return path to filename ``fn`` in the data folder.

    During testing it is often necessary to load data files. This
    function returns the full path to files in the ``data`` subfolder
    by default.

    Parameters
    ----------
    fn : str
        File name.
    subfolder : str, defaults to ``data``
        Name of the subfolder that contains the data.
    Returns
    -------
    str
        Inferred absolute path to the test data for the module where
        ``get_data_path(fn)`` is called.
    Notes
    -----
    The requested path may not point to an existing file, as its
    existence is not checked.

    This is from skbio's code base
    https://github.com/biocore/scikit-bio/blob/master/skbio/util/_testing.py#L50
    """
    # getouterframes returns a list of tuples: the second tuple
    # contains info about the caller, and the second element is its
    # filename
    callers_filename = inspect.getouterframes(inspect.currentframe())[1][1]
    path = os.path.dirname(os.path.abspath(callers_filename))
    data_path = os.path.join(path, subfolder, fn)
    return data_path
