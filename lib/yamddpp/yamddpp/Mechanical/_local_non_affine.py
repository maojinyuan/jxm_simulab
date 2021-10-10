from math import ceil

import numba as nb
import numpy as np
from numba import cuda

from ..utils import cu_mat_dot_v_pbc, cu_mat_dot_v


def local_non_affine_of_ab(sys_a, sys_b):
    gpu = sys_a.gpu  # all use sys_a device
    n_dim = sys_a.n_dim
    a0 = sys_a.d_pos_ortho
    b0 = sys_b.d_pos_ortho  # on the same device
    nl, nc = sys_a.d_nl, sys_a.d_nc
    n = sys_a.n
    Xij = np.zeros((n, n_dim, n_dim), dtype=np.float64)
    Yij = np.zeros((n, n_dim, n_dim), dtype=np.float64)
    DIV = np.zeros((n,), dtype=np.float64)
    d_Xij = cuda.to_device(Xij)
    d_Yij = cuda.to_device(Yij)
    d_DIV = cuda.to_device(DIV)

    # ndim: dimentsion, a.shape[1], jit does not support
    # using a.shape[1] directly in cuda.local.array creation,
    # numba==0.44.1
    # dim = np.ones(ndim, dtype=np.int64) * ndim
    # dimension of (-1, 0, 1) vector in calculating adjacent cells
    # a 3-D case would be (3, 3, 3), simply (3,) * n_d
    # _dim is required for in nonpython mode, sig=array(int64, 1d, A)
    # and using dim directly would by `readonly array(int64, 1d, C)'

    @cuda.jit(
        "void(float64[:,:],float64[:,:],float64[:],int64[:,:],int64[:],"
        "float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:])"
    )
    def _cu_XY(_a, _b, _box, _nl, _nc, _Xij, _Yij, _l0, _lt):
        i = cuda.grid(1)
        if i >= _a.shape[0]:
            return
        _dr0 = cuda.local.array(n_dim, nb.float64)
        _drt = cuda.local.array(n_dim, nb.float64)
        for j in range(_nc[i]):
            pj = _nl[i, j]
            cu_mat_dot_v_pbc(_l0, _a[pj], _a[i], _box, _dr0)
            cu_mat_dot_v_pbc(_lt, _b[pj], _b[i], _box, _drt)
            for k in range(n_dim):
                for l in range(n_dim):
                    _Xij[i, k, l] += _drt[k] * _dr0[l]
                    _Yij[i, k, l] += _dr0[k] * _dr0[l]

    @cuda.jit(
        "void(float64[:,:],float64[:,:],float64[:], int64[:,:],"
        "int64[:],float64[:,:,:], float64[:,:], float64[:,:], float64[:])"
    )
    def _cu_DIV(_a, _b, _box, _nl, _nc, _XIY, _l0, _lt, _ret):
        i = cuda.grid(1)
        if i >= _a.shape[0]:
            return
        _dr0 = cuda.local.array(n_dim, nb.float64)
        _drt = cuda.local.array(n_dim, nb.float64)
        _dr = cuda.local.array(n_dim, nb.float64)
        for j in range(_nc[i]):
            pj = _nl[i, j]
            cu_mat_dot_v_pbc(_l0, _a[pj], _a[i], _box, _dr0)
            cu_mat_dot_v_pbc(_lt, _b[pj], _b[i], _box, _drt)
            cu_mat_dot_v(_XIY[i], _dr0, _dr)
            for k in range(n_dim):
                _ret[i] += (_drt[k] - _dr[k]) ** 2
        if _nc[i] != 0:
            _ret[i] = _ret[i] / _nc[i]

    with cuda.gpus[gpu]:
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        bpg = ceil(n / tpb)
        _cu_XY[bpg, tpb](
            a0, b0, sys_a.d_box, nl, nc, Xij, Yij, sys_a.d_strain, sys_b.d_strain
        )
        # XIY = np.matmul(Xij, np.linalg.inv(Yij))
        d_Xij.copy_to_host(Xij)
        d_Yij.copy_to_host(Yij)
        cuda.synchronize()
        XIY = np.matmul(Xij, np.linalg.pinv(Yij, hermitian=True))
        d_XIY = cuda.to_device(XIY)
        # Moore-Penrose inverse, for nc[i] < ndim
        _cu_DIV[bpg, tpb](
            a0, b0, sys_a.d_box, nl, nc, d_XIY, sys_a.strain, sys_b.strain, d_DIV
        )
        d_DIV.copy_to_host(DIV)
        cuda.synchronize()
    return XIY, DIV
