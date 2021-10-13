import numpy as np
from ..utils import pbc
import numpy as np
from numba import cuda, float32, jit, int32

@cuda.jit('void(float32[:, :, :], int32[:], float32[:], float32, int32)')
def cu_rdf_kernel(_a, res, box, bs, bins):
    row, col = cuda.grid(2)
    if row >= _a.shape[0] or col >= _a.shape[1]:
        return

    xi, yi, zi = _a[row, col]
    for j in range(_a.shape[1]):
        if j > col:
            xj, yj, zj = _a[row, j] 
            dx = xj - xi
            dy = yj - yi
            dz = zj - zi
            dx -= box[0] * math.floor(dx/box[0]+0.5)
            dy -= box[1] * math.floor(dy/box[1]+0.5)
            dz -= box[2] * math.floor(dz/box[2]+0.5)
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            idx = int(math.floor(r / bs))
            if idx < bins:
                cuda.atomic.add(res, idx, 1)

def cu_rdf(a, box, bs, bins, gpu=0):
    TPB = 32 
    tpb = (TPB, TPB)
    bpg = (math.ceil(a.shape[0] / tpb[0]), math.ceil(a.shape[1] / tpb[1]))
    print("Threads per Block: (%d, %d) -- Blocks per Grid: (%d, %d)"%(tpb[0], tpb[1], bpg[0], bpg[1]))

    nt, na, nd = a.shape
    cuda.select_device(gpu)
    device = cuda.get_current_device()

    device_a   = cuda.to_device(a)
    device_res = cuda.to_device(np.zeros(bins, dtype=np.uint32))
    device_box = cuda.to_device(box)

    cu_rdf_kernel[bpg, tpb](device_a, device_res, device_box, bs, bins)
    cuda.synchronize()

    rho = na / (box[0] * box[1] * box[2])
    r = np.arange(bins) * bs
    vol = 4 * np.pi / 3 * ((r+bs)**3 - r**3)
    rr = device_res.copy_to_host() / na / nt / vol / rho
    cuda.close()

    return rr

def rdf_xy(x, y, box, dr=None, bins=None, use_gpu=False):
    """Calcualte radial distribution function
    :param x        : np.ndarray, positions
    :param y        : np.ndarray, positions
    :param dr       : float, gr spacing dr
    :param bins     : int, number of dr bins 
    :param use_gpu  : bool, calculate with gpu 

    todo: (1) bond/angle/mol exlcusions (2) fft (3) gpu (4) cufft
    """
    if dr   is None: dr   = 0.05
    if bins is None: bins = int(box.max() / 2 / dr)
    mode = 'xx' if x is y else 'xy' 

    rmax = dr * bins
    rho = x.shape[0] / box[0] / box[1] / box[2]
    if use_gpu is True:
        pass # todo
    else:
        rij = np.linalg.norm(pbc(x[:, None, :] - y[None, :, :], box), axis=-1)

    if mode == 'xx':
        rij = rij[np.triu_indices_from(rij, k=1)]
    hist, edges = np.histogram(rij, bins=bins, range=(0, rmax))
    vol = 4 * np.pi / 3.0 * (edges[1:]**3 - edges[:-1]**3)
    gr = mode.count('x') * hist / rho / vol / x.shape[0]
    r_mid = edges[1:] - dr / 2

    return np.c_[r_mid, gr]

if __name__ == '__main__':
    pass
