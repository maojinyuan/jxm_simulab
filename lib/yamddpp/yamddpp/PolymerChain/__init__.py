import numpy as np
from numba import float64
from numba import guvectorize

from ._bond_angle import bond_angles
from ._bond_angle import bond_angles_ufunc
from ._rg_tensor import batch_rg_tensor
from ._rouse_modes import normal_modes
from ..Aggregation import pbc


def bond_vecs_common(samples, boxes):
    if samples.ndim == 2:
        samples = np.expand_dims(samples, 0)
    if samples.ndim < boxes.ndim + 2:
        raise ValueError(
            "Are you using multiple box values for an 1-frame sample?"
        )
    # only 2 dimensions are allowed:
    # samples: (..., n_batch, n_frames, N_CHAINS, CHAIN_LENGTH, n_dim)
    # boxes: (..., n_batch, n_frames, n_dim)
    boxes = np.expand_dims(np.expand_dims(boxes, -2), -3)
    return samples, boxes


def bond_vecs(samples: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    r"""Batch calculation of bond vectors of linear polymers
    :param samples: np.ndarray, (...,n_chains, n_monomers,n_dim)
    e.g. (n_batch, n_frames, n_chains, n..., n_dim)
    :param boxes: np.ndarray, (...,n_dimensions),
    e.g. (n_batch, n_frames, n_dim)
    :return: np.ndarray ret. (..., n_chains, chain_length, n_dim) with
    (0, ...) for the 1st monomer on each chain.
    """
    samples, boxes = bond_vecs_common(samples, boxes)
    return pbc(
        np.diff(samples, axis=-2, prepend=samples[..., :1, :]), boxes
    )


@guvectorize([(float64[:, :], float64[:, :], float64[:, :])],
             '(n,p),(p,m)->(n,m)', target='parallel')  # target='cpu','gpu'
def batch_dot(a, b, ret):  # much more faster than np.tensordot or np.einsum
    r"""Vectorized universal function.
    :param a: np.ndarray, (...,N,P)
    :param b: np.ndarray, (...,P,M)
    axes will be assigned automatically to last 2 axes due to the signatures.
    this functions is actual np.einsum('...mp,....pn->...mn', a, b), or
    np.matmul(a, b). But this is much faster.
    :param ret: np.ndarray, generated automatically by guvectorize
    :return: np.ndarray, results. (...,N,M)
    """
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            tmp = 0.
            for k in range(a.shape[1]):
                tmp += a[i, k] * b[k, j]
            ret[i, j] = tmp


@guvectorize([(float64[:], float64[:], float64[:])],
             '(n),(n)->()', targer='parallel')
def batch_inner_prod(a, b, ret):
    tmp1 = tmp2 = tmp3 = 0
    for i in range(a.shape[0]):
        tmp1 += a[i] * b[i]
        tmp2 += a[i] * a[i]
        tmp3 += b[i] * b[i]
    ret[0] = tmp1 / (tmp2 * tmp3) ** 0.5

# TODO: CReTA
