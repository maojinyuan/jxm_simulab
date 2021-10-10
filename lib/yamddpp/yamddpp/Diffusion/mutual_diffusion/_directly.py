import math

import numpy as np
from numba import cuda


# Mutual mean square displacements: \sum_{ij}\langle|r_i(t)-r_j(0)|^2\rangle


@cuda.jit("float64(float64[:], float64[:])", device=True)
def distance2(a, b):
    tmp = 0
    for i in range(a.shape[0]):
        tmp += (a[i] - b[i]) ** 2
    return tmp


@cuda.jit("void(float64[:,:,:], float64[:,:])")
def _cu_kernel(x, ret):  # ret -> (n_particles, n_frames)
    i, j = cuda.grid(2)  # x -> (n_frames, n_particles, n_dim)
    if i >= x.shape[0]:
        return
    if j >= x.shape[0] - i:
        return
    for k in range(x.shape[1] - 1):
        pkt = x[i + j, k]
        for l in range(k, x.shape[1]):
            pl0 = x[i, l]
            dr2 = distance2(pkt, pl0)
            cuda.atomic.add(ret[k], j, dr2)
            cuda.atomic.add(ret[l], j, dr2)


@cuda.jit("void(float64[:,:,:], float64[:])")
def _cu_kernel_cum(x, ret):  # ret -> (n_frames,)
    i, j = cuda.grid(2)
    if i >= x.shape[0]:
        return
    if j >= x.shape[0] - i:
        return
    for k in range(x.shape[1] - 1):
        pkt = x[i + j, k]
        for l in range(k + 1, x.shape[1]):
            pl0 = x[i, l]
            dr2 = distance2(pkt, pl0)
            cuda.atomic.add(ret, j, 2 * dr2)


def cu_mutual_diffusion(x, cum=True, gpu=0):
    r"""Mutual diffusion calculation.
    :param x: np.ndarray, nd-coordinates with (n_frame, n_particles, n_dimension)
    :param cum: bool, summing over n_particles or not
    :param gpu: int, choose gpu
    :return: np.ndarray, mutual MSD
    """
    x = x.astype(np.float64)
    with cuda.gpus[gpu]:
        device = cuda.get_current_device()
        tpb = (device.WARP_SIZE,) * 2
        bpg = (math.ceil(x.shape[0] / tpb[0]),
               math.ceil(x.shape[0] / tpb[1]))
        if not cum:
            ret = np.zeros((x.shape[1], x.shape[0]),
                           dtype=np.float64)
            _cu_kernel[bpg, tpb](x, ret)
            ret = ret.T
        else:
            ret = np.zeros(x.shape[0], dtype=np.float64)
            _cu_kernel_cum[bpg, tpb](x, ret)
    return ret  # ret -> (n_frames, n_particles) or (n_frames,)
