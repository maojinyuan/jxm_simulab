#!/usr/bin/env python

import numpy as np
from numba import cuda, float32, jit, int32
import math

def autocorrFFT(x):
	M=X.shape[0]
	Fx = np.fft.rfft(X.T[0], 2*M)
	Fy = np.fft.rfft(X.T[1], 2*M)
	Fz = np.fft.rfft(X.T[2], 2*M)
	corr = abs(Fx)**2 + abs(Fy)**2 + abs(Fz)**2
	res = np.fft.irfft(corr)
	res= (res[:M]).real
	return res

def msd_fft(x):
     """ FFT version: calculate msd
    
    :param x : np.ndarray, ndim = 2, trajectory of one particle in 3D.
    :return  : np.ndarray
    """
    
	M = X.shape[0]
	D = np.square(X).sum(axis=1)
	D = np.append(D, 0)
	S2 = autocorrFFT(X)
	S2 = S2 / np.arange(M, 0, -1)
	Q = 2 * D.sum()
	S1 = np.zeros(M)
	for m in range(M):
		Q = Q - D[m-1] -D[M-m]
		S1[m] = Q / (M-m)
	return S1 - 2 * S2


def msd_numpy(_a):
    """ Numpy version: calculate msd
    
    :param _a: np.ndarray, ndim = 3
    :return  : np.ndarray
    """
    
    return np.asarray([((_a[inv:] - _a[:-inv])**2).sum(axis=-1).mean() if inv != 0 else 0.0 for inv in range(_a.shape[0])], dtype=np.float32)

@cuda.jit('void(float32[:, :, :], float32[:, :])')
def msd_cu(_a, _r2):
    """ Numba version: calculate msd
    
    :param _a: np.ndarray, ndim = 3, trajectory with multi particles in 3D.
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

    msd_cu[bpg, tpb](device_a, device_r2)
    cuda.synchronize()

    res_r2 = device_r2.copy_to_host()
    cuda.close()

    return res_r2
