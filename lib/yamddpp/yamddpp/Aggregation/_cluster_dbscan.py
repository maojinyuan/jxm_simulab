from math import ceil

import numpy as np
from numba import cuda
from sklearn.cluster import DBSCAN

from . import pbc_pairwise_distance


def cluster(pos, box, weights=None, epsilon=1.08,
            minpts=5, gpu=0):  # these parameters are suitable for dpd with density = 3.0
    r"""
    :param pos: np.ndarray, position
    :param box: np.ndarray, box
    :param weights: np.ndarray, weights
    :param epsilon: float
    :param minpts: int
    :param gpu: int
    :return: np.ndarray, labels[i] is which cluster which contains pos[i].
    """
    ret = np.zeros((pos.shape[0],) * 2, dtype=np.float)
    device = cuda.get_current_device()
    tpb2d = (device.WARP_SIZE,) * 2
    bpg2d = (ceil(pos.shape[0] / tpb2d[0]), ceil(pos.shape[0] / tpb2d[1]))
    with cuda.gpus[gpu]:
        pbc_pairwise_distance[bpg2d, tpb2d](pos, pos, box, ret)
    db_fitted = DBSCAN(
        metric='precomputed', n_jobs=-1, eps=epsilon, min_samples=minpts
    ).fit(
        ret, sample_weight=weights
    )
    # clusters = [pos[db_fitted.labels_ == _]
    #            for _ in list(set(db_fitted.labels_)) if _ != -1]
    # noises = pos[db_fitted.labels_ == -1]
    return db_fitted.labels_
