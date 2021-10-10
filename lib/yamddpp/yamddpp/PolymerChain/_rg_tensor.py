import numpy as np

from . import batch_dot
from . import bond_vecs_common
from ..Aggregation import pbc


def batch_rg_tensor(x: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    r"""Batch calculation of Rg Tensors.
    :param x: np.ndarray, (...,n_chains, n_monomers,n_dim)
    e.g. (n_batch, n_frames, n_chains, n..., n_dim)
    :param boxes: np.ndarray, (...,n_dimensions),
    e.g. (n_batch, n_frames, n_dim)
    :return: np.ndarray ret.
    """
    x, boxes = bond_vecs_common(x, boxes)
    bv = pbc(
        np.diff(x, axis=-2, prepend=x[..., :1, :]), boxes
    ).cumsum(axis=-2)
    ree = bv[..., -1, :]
    cm = np.expand_dims(bv.mean(axis=-2), -2)
    bv = bv - cm
    cm = pbc(x[..., :1, :] + cm, boxes)
    return batch_dot(np.swapaxes(bv, -2, -1), bv) / bv.shape[-2], cm, ree
