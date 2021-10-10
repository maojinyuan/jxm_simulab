import numpy as np

from . import batch_inner_prod
from . import bond_vecs


def bond_angles(samples: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    r"""Batch calculation of bond angles of linear polymers.
    :param samples: np.ndarray, (...,n_chains, n_monomers,n_dim)
    e.g. (n_batch, n_frames, n_chains, n..., n_dim)
    :param boxes: np.ndarray, (...,n_dimensions),
    e.g. (n_batch, n_frames, n_dim)
    :return: np.ndarray ret. (..., n_chains, chain_length - 2, n_dim)
    """
    bonds = bond_vecs(samples, boxes)
    # 1st bond vec is 0. Angles start from r2 - r1
    prod = np.einsum(
        '...ij,...ij->...i', bonds[..., 1:-1, :], bonds[..., 2:, :]
    )
    norm = np.linalg.norm(bonds, axis=-1)
    return np.arccos(np.clip(prod / norm[..., 1:-1] / norm[..., 2:], -1, 1))


def bond_angles_ufunc(samples: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    r"""Batch calculation of bond angles of linear polymers. Using ufunc to accelerate.
    :param samples: np.ndarray, (...,n_chains, n_monomers,n_dim)
    e.g. (n_batch, n_frames, n_chains, n..., n_dim)
    :param boxes: np.ndarray, (...,n_dimensions),
    e.g. (n_batch, n_frames, n_dim)
    :return: np.ndarray ret. (..., n_chains, chain_length - 2, n_dim)
    This version should be faster. ^_^
    """
    bonds = bond_vecs(samples, boxes)
    prod = batch_inner_prod(bonds[..., 1:-1, :], bonds[..., 2:, :])
    return np.arccos(np.clip(prod, -1, 1))
