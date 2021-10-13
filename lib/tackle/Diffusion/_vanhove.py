import numpy as np
from numba import cuda, float32, jit, int32
import math


@cuda.jit('void(float32[:, :, :], float32[:], int64, float32, float32)')
def cu_vanhove(_a, _res, _inv, _bs, _maxbin):
    row, col = cuda.grid(2)
    if row >= _a.shape[0] or col >= _a.shape[1]:
        return

    xi, yi, zi = _a[row, col]
    j = row + _inv 
    if j < _a.shape[0]:
        xj, yj, zj = _a[j, col] 
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        r2 = dx * dx + dy * dy + dz * dz
        r = math.sqrt(r2)
        idx = math.floor(r / _bs) 
        if idx < _maxbin:
            cuda.atomic.add(_res, idx, 1)
        
def _call(a, tinv, bs, maxbin, gpu=0):
    TPB = 32 
    tpb = (TPB, TPB)
    bpg = (math.ceil(a.shape[0] / tpb[0]), math.ceil(a.shape[1] / tpb[1]))

    nt, na, nd = a.shape
    cuda.select_device(gpu)
    device = cuda.get_current_device()

    device_res = cuda.device_array(maxbin, dtype=np.float32) 
    device_res.copy_to_device(np.zeros(maxbin, dtype=np.float32)) 

    device_a = cuda.device_array(a.shape, dtype=np.float32) 
    device_a.copy_to_device(a)

    cu_vanhove[bpg, tpb](device_a, device_res, tinv, bs, maxbin)
    cuda.synchronize()

    res = device_res.copy_to_host()

    cuda.close()

    return res
