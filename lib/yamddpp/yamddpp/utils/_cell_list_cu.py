from math import ceil
from math import floor

import numpy as np
from numba import cuda


@cuda.jit("int64(float64[:], float64[:], int64[:])", device=True)
def cu_cell_id(p, box, ibox):
    ret = floor((p[0] / box[0] + 0.5) * ibox[0])
    tmp = ibox[0]
    for i in range(1, p.shape[0]):
        ret += floor((p[i] / box[i] + 0.5) * ibox[i]) * tmp
        tmp *= ibox[i]
    return ret
    # return floor((p[0] / box[0] + 0.5) * ibox[0]) + \
    # floor((p[1] / box[1] + 0.5) * ibox[1]) * ibox[0] + \
    # floor((p[2] / box[2] + 0.5) * ibox[2]) * ibox[1] * ibox[0]
    # +0.5 for 0 is at center of box.
    # unravel in Fortran way.


@cuda.jit("void(float64[:, :], float64[:], int64[:], int64[:])")
def cu_cell_ind(pos, box, ibox, ret):
    i = cuda.grid(1)
    if i < pos.shape[0]:
        pi = pos[i]
        ic = cu_cell_id(pi, box, ibox)
        ret[i] = ic


def cell_count(cell_id, n_cell):
    r"""
    :param cell_id: must be sorted.
    :return: cumsum of count of particles in cells.
    """
    return np.append(0, np.cumsum(np.bincount(cell_id, minlength=n_cell)))


@cuda.jit("void(int64[:], int64[:])")
def cu_cell_count(cell_id, ret):
    i = cuda.grid(1)
    if i >= cell_id.shape[0]:
        return
    cuda.atomic.add(ret, cell_id[i] + 1, 1)


def cu_cell_list_argsort(pos, box, ibox, gpu=0):
    n = pos.shape[0]
    n_cell = np.multiply.reduce(ibox)
    cell_id = np.zeros(n).astype(np.int64)
    with cuda.gpus[gpu]:
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        bpg = ceil(n / tpb)
        cu_cell_ind[bpg, tpb](pos, box, ibox, cell_id)
    cell_list = np.argsort(cell_id)  # pyculib radixsort for cuda acceleration.
    cell_id = cell_id[cell_list]
    cell_counts = np.r_[0, np.cumsum(np.bincount(cell_id, minlength=n_cell))]
    return cell_list.astype(np.int64), cell_counts.astype(np.int64)

# calling: for a particle in nth cell, cell_count[n] gives the start index of
# cell_list, and cell_count[n+1] gives the end index of cell_list.
# e.g. for 0th cell, cell_count[0] = 0 and cell_count[1] = number_of_particles_in_0th_cell
# for ith cell, cell_count[i] = number_of_particles_of_all_(0, i-1)th_cell and cell_count[i+1]
# = number_of_particles_of_all_(0, i)th_cell. For cell_list is sorted by cell_ids:
# cell_list = [pid_in_0th_cell, pid_in_0th_cell, ..., pid_in_nth_cell]
# cell_list[number_of_particles_of_all_(0, i-1)th_cell] = 1st_pid_in_ith_cell
# algorithm: http://developer.download.nvidia.com/GTC/PDF/2062_Wang.pdf
