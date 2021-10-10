from math import ceil

import numba as nb
import numpy as np
from numba import cuda

from ._cell_list_cu import cu_cell_id
from ._cell_list_cu import cu_cell_list_argsort
from ._utils import add_local_arr_mois_1
from ._utils import cu_max_int
from ._utils import cu_set_to_int
from ._utils import pbc_dist_cu
from ._utils import ravel_index_f_cu
from ._utils import unravel_index_f_cu


def cu_nl(a, box, rc, nc_p=100, gpu=0):
    dim = np.ones(a.shape[1], dtype=np.int64) * 3
    ndim = a.shape[1]
    ibox = np.asarray(np.round(box / rc), dtype=np.int64)
    cl, cc = cu_cell_list_argsort(a, box, ibox, gpu=gpu)

    # neighbour is always smaller than max_cell * 3 ** ndim
    # ret = np.zeros((a.shape[0], cc.max() * 3 ** ndim), dtype=np.int64)
    # nc = np.zeros((a.shape[0],), dtype=np.int64)

    @cuda.jit(
        "void(float64[:,:],float64[:],int64[:],"
        "float64,int64[:],int64[:],int64[:,:],int64[:], int64[:])"
    )
    def _nl(_a, _box, _ibox, _rc, _cl, _cc, _ret, _nc, _dim):
        r"""
        :param _a: positions of a, (n_pa, n_d)
        :param _box: box, (n_d,)
        :param _ibox: bins, (n_d,)
        :param _rc: r_cut of neighbour, float64
        :param _cl: cell-list of b, (n_pb,)
        :param _cc: cell-count-cum, (n_cell + 1,)
        :param _ret: neighbour list (n_pa, nn)
        :param _nc: neighbour count (n_pa,)
        :param _dim: dimension
        :return: None
        """
        i = cuda.grid(1)
        if i >= _a.shape[0]:
            return
        cell_i = cu_cell_id(_a[i], _box, _ibox)  # a[i] in which cell
        cell_vec_i = cuda.local.array(ndim, nb.int64)  # unravel the cell id
        unravel_index_f_cu(cell_i, _ibox, cell_vec_i)  # unravel the cell id
        cell_vec_j = cuda.local.array(ndim, nb.int64)
        for j in range(3 ** ndim):  # 3**ndim cells
            unravel_index_f_cu(j, _dim, cell_vec_j)
            add_local_arr_mois_1(cell_vec_j, cell_vec_i)
            # cell_vec_i + (-1, -1, -1) to (+1, +1, +1)
            # unraveled results would be (0,0,0) to (2,2,2) for dim=3
            cell_j = ravel_index_f_cu(cell_vec_j, _ibox)  # ravel cell id vector to cell id
            start = _cc[cell_j]  # start pid in the cell_j th cell
            end = _cc[cell_j + 1]  # end pid in the cell_j th cell
            for k in range(start, end):  # particle ids in cell_j
                pid_k = _cl[k]
                dr = pbc_dist_cu(_a[pid_k], _a[i], _box)
                if dr < _rc:
                    if _nc[i] < _ret.shape[1]:
                        _ret[i, _nc[i]] = pid_k
                    _nc[i] += 1

    with cuda.gpus[gpu]:
        # begin from a guess, if max(nc) < nc_p, the nl should be run only once.
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        bpg = ceil(a.shape[0] / tpb)
        d_nc = cuda.device_array((a.shape[0],), dtype=np.int64)
        d_nl = cuda.device_array((a.shape[0], nc_p), dtype=np.int64)
        d_nc_max = cuda.device_array((1,), dtype=np.int64)
        while True:
            cu_set_to_int[bpg, tpb](d_nc, 0)
            _nl[bpg, tpb](
                a, box, ibox, rc, cl, cc, d_nl, d_nc, dim
            )
            cu_max_int(d_nc, d_nc_max)
            nc_max = d_nc_max.copy_to_host()
            if nc_max[0] > nc_p:
                nc_p = nc_max[0]
                d_nl = cuda.device_array((a.shape[0], nc_p), dtype=np.int64)
            else:
                break
    # return nc_p, as the initial guess for next calculation
    return d_nl, d_nc, nc_p


def cu_nl_strain(n_dim):
    # dim = np.ones(n_dim, dtype=np.int64) * 3
    # neighbour is always smaller than max_cell * 3 ** ndim
    # ret = np.zeros((a.shape[0], cc.max() * 3 ** ndim), dtype=np.int64)
    # nc = np.zeros((a.shape[0],), dtype=np.int64)

    @cuda.jit(
        "void(float64[:,:],float64[:],int64[:],float64[:,:]"
        "float64,int64[:],int64[:],int64[:,:],int64[:], int64[:])"
    )
    def _nl(_a, _box, _ibox, _str, _rc, _cl, _cc, _ret, _nc, _dim):
        r"""
        :param _a: positions of a, (n_pa, n_d)
        :param _box: box, (n_d,)
        :param _ibox: bins, (n_d,)
        :param _rc: r_cut of neighbour, float64
        :param _cl: cell-list of b, (n_pb,)
        :param _cc: cell-count-cum, (n_cell + 1,)
        :param _ret: neighbour list (n_pa, nn)
        :param _nc: neighbour count (n_pa,)
        :param _dim: dimension
        :return: None
        """
        i = cuda.grid(1)
        if i >= _a.shape[0]:
            return
        cell_i = cu_cell_id(_a[i], _box, _ibox)  # a[i] in which cell
        cell_vec_i = cuda.local.array(n_dim, nb.int64)  # unravel the cell id
        unravel_index_f_cu(cell_i, _ibox, cell_vec_i)  # unravel the cell id
        cell_vec_j = cuda.local.array(n_dim, nb.int64)
        for j in range(3 ** n_dim):  # 3**ndim cells
            unravel_index_f_cu(j, _dim, cell_vec_j)
            add_local_arr_mois_1(cell_vec_j, cell_vec_i)
            # cell_vec_i + (-1, -1, -1) to (+1, +1, +1)
            # unraveled results would be (0,0,0) to (2,2,2) for dim=3
            cell_j = ravel_index_f_cu(cell_vec_j, _ibox)  # ravel cell id vector to cell id
            start = _cc[cell_j]  # start pid in the cell_j th cell
            end = _cc[cell_j + 1]  # end pid in the cell_j th cell
            for k in range(start, end):  # particle ids in cell_j
                pid_k = _cl[k]
                # dr = pbc_dist_cu(_a[pid_k], _a[i], _box)
                # use ortho-pos here and apply strain on dr
                if i == pid_k:
                    continue
                # dr = cu_mat_dot_v_pbc_dist(_str, _a[pid_k], _a[i], _box)
                dr = pbc_dist_cu(_a[pid_k], _a[i], _box)
                # use ortho-box _box, distance can be directly calculated.
                if dr < _rc:
                    if _nc[i] < _ret.shape[1]:
                        _ret[i, _nc[i]] = pid_k
                    _nc[i] += 1
