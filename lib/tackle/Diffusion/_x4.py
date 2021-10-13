import numpy as np
from numba import cuda, float32, jit, int32
import math


@cuda.jit('void(float32[:, :, :], float32[:], int64[:])')
def cu_xm_x4(_a, _x4, _num):
    row, col = cuda.grid(2)
    if row >= _a.shape[0] or col >= _a.shape[1]:
        return

    xi, yi, zi = _a[row, col]
    for j in range(_a.shape[0]):
        inv = row - j
        if row > j and inv != 0:
            xj, yj, zj = _a[j, col] 
            dx = xj - xi
            dy = yj - yi
            dz = zj - zi
            index = _num[inv - 1] + j
            r2 = dx * dx + dy * dy + dz * dz

            # x4
            r = math.sqrt(r2)
            if r < 0.3:
                cuda.atomic.add(_x4, index, 1)

def _call(a, gpu=0):
    TPB = 32 
    tpb = (TPB, TPB)
    bpg = (math.ceil(a.shape[0] / tpb[0]), math.ceil(a.shape[1] / tpb[1]))
    print("Threads per Block: (%d, %d) -- Blocks per Grid: (%d, %d)"%(tpb[0], tpb[1], bpg[0], bpg[1]))

    nt, na, nd = a.shape
    cuda.select_device(gpu)
    device = cuda.get_current_device()

    ndata = int(np.arange(1, nt + 1).sum())
    index_num = np.arange(1, nt+1)[::-1].cumsum()
    num = cuda.device_array(index_num.shape[0], dtype=np.int) 
    num.copy_to_device(index_num)

    #  device_xm = cuda.device_array(ndata, dtype=np.float32) 
    #  device_xm.copy_to_device(np.zeros(ndata, dtype=np.float32))

    device_x4 = cuda.device_array(ndata, dtype=np.float32) 
    device_x4.copy_to_device(np.zeros(ndata, dtype=np.float32))

    device_a = cuda.device_array(a.shape, dtype=np.float32) 
    device_a.copy_to_device(a)

    # deallocate pos in host
    pos = None
    a = None

    #  cu_xm_x4[bpg, tpb](device_a, device_xm, device_x4, num)
    cu_xm_x4[bpg, tpb](device_a, device_x4, num)
    cuda.synchronize()

    #  res_xm = device_xm.copy_to_host()
    res_x4 = device_x4.copy_to_host()

    cuda.close()

    #  return res_xm, res_x4
    return res_x4
        
