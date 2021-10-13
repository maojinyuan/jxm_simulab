import numpy as np
from numba import cuda, float32, jit, int32
import math

@cuda.jit('void(float32[:, :, :], float32[:, :], float32[:, :])')
def cu_alpha2(_a, _r2, _r4):
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
            cuda.atomic.add(_r4, (inv, col), r4 / num)

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

    device_r4 = cuda.device_array((nt, na), dtype=np.float32) 
    device_r4.copy_to_device(np.zeros((nt, na), dtype=np.float32))

    device_a = cuda.device_array(a.shape, dtype=np.float32) 
    device_a.copy_to_device(a)

    # deallocate pos in host
    pos = None
    a = None

    cu_alpha2[bpg, tpb](device_a, device_r2, device_r4)
    cuda.synchronize()

    res_r2 = device_r2.copy_to_host()
    res_r4 = device_r4.copy_to_host()
    cuda.close()

    return res_r2, res_r4
        
