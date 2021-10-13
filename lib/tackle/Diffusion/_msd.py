#!/usr/bin/env python

import numpy as np
from numba import cuda, float32, jit, int32
import math


def msd_numpy(_a):
    """ Numpy version: calculate msd
    
    :param _a: np.ndarray, ndim = 3
    :return  : np.ndarray
    """
    
    return np.asarray([((_a[inv:] - _a[:-inv])**2).sum(axis=-1).mean() if inv != 0 else 0.0 for inv in range(_a.shape[0])], dtype=np.float32)

@cuda.jit('void(float32[:, :, :], float32[:, :])')
def cu_msd(_a, _r2):
    """ Numba version: calculate msd
    
    :param _a: np.ndarray, ndim = 3, trajectory with multi particles in 3-dimensions
    :return  : np.ndarray, ndim = 2 
    """
    row, col = cuda.grid(2)
    if row >= _a.shape[0] or col >= _a.shape[1]:
        return

    xi, yi, zi = _a[row, col]
    for j in range(_a.shape[0]):
        if row > j:
            xj, yj, zj = _a[j, col] 
            dx = xj - xi
            dy = yj - yi
            dz = zj - zi
            inv = row - j
            num = _a.shape[0] - inv
            r2 = dx * dx + dy * dy + dz * dz
            r4 = r2**2
            cuda.atomic.add(_r2, (inv, col), r2 / num)

def _call(a, gpu=0):
    TPB = 32 
    tpb = (TPB, TPB)
    bpg = (math.ceil(a.shape[0] / tpb[0]), math.ceil(a.shape[1] / tpb[1]))
    print("Threads per Block: (%d, %d) -- Blocks per Grid: (%d, %d)"%(tpb[0], tpb[1], bpg[0], bpg[1]))

    nt, na, nd = a.shape

    cuda.select_device(gpu)
    device = cuda.get_current_device()

    device_r2 = cuda.device_array((nt, na), dtype=np.float32) 
    device_r2.copy_to_device(np.zeros((nt, na), dtype=np.float32))

    device_a = cuda.device_array(a.shape, dtype=np.float32) 
    device_a.copy_to_device(a)

    # deallocate pos in host
    pos = None
    a = None

    cu_msd[bpg, tpb](device_a, device_r2)
    cuda.synchronize()

    res_r2 = device_r2.copy_to_host()
    cuda.close()

    return res_r2
