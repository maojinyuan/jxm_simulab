import math
import sys
import numba as nb
import numpy as np
from numba import cuda


@cuda.jit("float64(float64[:], float64[:])", device=True)
def dot(a, b):
    tmp = 0
    for i in range(a.shape[0]):
        tmp += a[i] * b[i]
    return tmp


@cuda.jit("void(float64[:,:,:], float64[: ,:],  float64[:, :])")  # ret -> (n_particle,  n_frames)
def _cu_kernel(traj, qs, ret):
    i, j = cuda.grid(2)
    if i >= traj.shape[0]:
        return
    if j >= traj.shape[0] - i:
        return
    drk = cuda.local.array(3, nb.float64)
    for k in range(traj.shape[1]):
        for t in range(traj.shape[2]):
            drk[t] = traj[i + j, k, t] - traj[i, k, t]
        for l in range(qs.shape[0]):
            qvec = qs[l]
            cuda.atomic.add(ret[k], j, math.cos(-dot(qvec, drk)))


def incoherent_scattering(traj, q, dq, q_vectors=None):
    ret = np.zeros((traj.shape[1], traj.shape[0]))
    n = int(np.round(q / dq))
    q_vecs = q_vectors
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>!!!")
    if q_vecs is None:
        @nb.jit(nopython=True)
        def _generate_q_vecs(_n, _dq, _q, _shape):
            ret = []
            for _qq in np.ndindex(_shape):
                q_tmp = 0
                for _qqq in _qq:
                    q_tmp += (_qqq - _n) ** 2
                q_tmp = q_tmp**0.5 * _dq
                if abs(q_tmp - _q) / _q < 1.5e-3:
                    ret.append(_qq)
            return ret

        shape = (n * 2,) * 3
        q_vecs = _generate_q_vecs(n, dq, q, shape)
        q_vecs = np.asarray((np.array(q_vecs) - n) * dq, dtype=np.float64)
    print('Start with Q vecs:', q_vecs.shape)
    import time
    s = time.time()
    with cuda.gpus[2]:
        device = cuda.get_current_device()
        tpb = (device.WARP_SIZE,) * 2
        bpg = (math.ceil(traj.shape[0] / tpb[0]),
               math.ceil(traj.shape[0] / tpb[1]))
        _cu_kernel[bpg, tpb](traj, q_vecs, ret)
    print(time.time() - s)
    return ret.T / np.arange(traj.shape[0], 0, -1)[:, None] / q_vecs.shape[0], q_vecs
