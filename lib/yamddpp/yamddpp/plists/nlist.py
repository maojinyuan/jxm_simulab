import numpy as np
from numba import cuda, void, int32, float32, float64

from .clist import clist
from ..utils import cu_set_to_int


@cuda.jit(void(int32[:], int32[:]))
def cu_max_int(arr, arr_max):
    i = cuda.grid(1)
    if i >= arr.shape[0]:
        return
    cuda.atomic.max(arr_max, 0, arr[i])


def _gen_func(dtype):
    from math import floor
    float = float64
    if dtype == np.dtype(np.float32):
        float = float32

    @cuda.jit(float(float[:], float[:], float[:]), device=True)
    def cu_pbc_dist2(a, b, box):
        ret = 0
        for i in range(a.shape[0]):
            d = a[i] - b[i]
            d -= box[i] * floor(d / box[i] + 0.5)
            ret += d ** 2
        return ret

    @cuda.jit(
        void(float[:, :], float[:], float, int32[:, :], int32[:, :], int32[:], int32[:],
             int32[:, :], int32[:], int32[:], int32))
    def cu_nlist(x, box, r_cut2, cell_map, cell_list, cell_count, cells, nl, nc, n_max, contain_self):
        pi = cuda.grid(1)
        if pi >= x.shape[0]:
            return
        # xi = cuda.local.array(ndim, dtype=float64)
        # xj = cuda.local.array(ndim, dtype=float64)
        # for l in range(ndim):
        #    xi[l] = x[pi, l]
        ic = cells[pi]
        n_needed = 0
        nn = 0
        xi = x[pi]
        for j in range(cell_map.shape[1]):
            jc = cell_map[ic, j]
            for k in range(cell_count[jc]):
                pj = cell_list[jc, k]
                if contain_self == 0 and pj == pi:  # == 0 means do not contain self.
                    continue
                # for m in range(ndim):
                # xj[m] = x[pj, m]
                r2 = cu_pbc_dist2(xi, x[pj], box)
                if r2 < r_cut2:
                    if nn < nl.shape[1]:
                        nl[pi, nn] = pj
                    else:
                        n_needed = nn + 1
                    nn += 1
        nc[pi] = nn
        if nn > 0:
            cuda.atomic.max(n_max, 0, n_needed)

    return cu_nlist


class nlist(object):
    def __init__(self, frame, cell_guess=50, n_guess=150, contain_self=0):
        self.frame = frame
        self.n = self.frame.n
        self.box = frame.box
        self.cell_guess = cell_guess
        self.n_guess = n_guess
        self.gpu = frame.gpu
        self.tpb = 64
        self.bpg = int(frame.n // self.tpb + 1)
        self.contain_self = contain_self
        # self.situ_zero = np.zeros(1, dtype=np.int32)
        self.update_counts = 0
        self.cu_nlist = _gen_func(frame.x.dtype)
        with cuda.gpus[self.gpu]:
            #self.p_n_max = cuda.pinned_array((1,), dtype=np.int32)
            self.d_n_max = cuda.device_array(1, dtype=np.int32)
            self.d_nl = cuda.device_array((self.frame.n, self.n_guess), dtype=np.int32)
            self.d_nc = cuda.device_array((self.frame.n,), dtype=np.int32)
            self.d_situation = cuda.device_array(1, dtype=np.int32)
        self.clist = clist(self.frame, cell_guess=self.cell_guess)
        self.neighbour_list()

    def neighbour_list(self):
        with cuda.gpus[self.gpu]:
            if self.n != self.frame.n:
                self.bpg = int(self.frame.n // self.tpb + 1)
                self.d_nl = cuda.device_array((self.frame.n, self.n_guess), dtype=np.int32)
                self.d_nc = cuda.device_array((self.frame.n,), dtype=np.int32)
            while True:
                cu_set_to_int[self.bpg, self.tpb](self.d_nc, 0)
                # reset situation while build nlist
                self.cu_nlist[self.bpg, self.tpb](self.frame.d_x,
                                                  self.frame.d_box,
                                                  self.frame.r_cut2,
                                                  self.clist.d_cell_map,
                                                  self.clist.d_cell_list,
                                                  self.clist.d_cell_counts,
                                                  self.clist.d_cells,
                                                  self.d_nl,
                                                  self.d_nc,
                                                  self.d_n_max,
                                                  self.contain_self)
                p_n_max = self.d_n_max.copy_to_host()
                cuda.synchronize()
                # n_max = np.array([120])
                if p_n_max[0] > self.n_guess:
                    self.n_guess = p_n_max[0]
                    self.n_guess = self.n_guess + 8 - (self.n_guess & 7)
                    self.d_nl = cuda.device_array((self.frame.n, self.n_guess), dtype=np.int32)
                else:
                    break
        self.n = self.frame.n

    def update(self):
        self.clist.update()
        self.neighbour_list()
        self.update_counts += 1

    def show(self):
        cell_list = self.clist.d_cell_list.copy_to_host()
        cell_map = self.clist.d_cell_map.copy_to_host()
        cell_counts = self.clist.d_cell_counts.copy_to_host()
        nl = self.d_nl.copy_to_host()
        nc = self.d_nc.copy_to_host()
        cuda.synchronize()
        return cell_list, cell_counts, cell_map, nl, nc
