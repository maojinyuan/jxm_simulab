import numpy as np
from numba import jit
from numpy.core.multiarray import ndarray


@jit
def cell_id(x: ndarray, box: ndarray, ibox: ndarray) -> int:
    ind: ndarray = np.asarray(((x / box + 0.5) * ibox), dtype=np.int64)
    return ind[0] + ind[1] * ibox[0] + ind[2] * ibox[1] * ibox[0]


@jit
def i_cell(cid: ndarray, ibox: ndarray) -> int:  # ravel in Fortran way.
    ind: ndarray = np.asarray((cid + ibox) % ibox, dtype=np.int64)
    return ind[0] + ind[1] * ibox[0] + ind[2] * ibox[1] * ibox[0]


# TODO: all-dimension support for above 2 funcs.


@jit
def unravel_index_f(i, dim):  # unravel index in Fortran way.
    dim = np.asarray(dim)
    ret = np.zeros(dim.shape)
    for k in range(dim.shape[0]):
        ret[k] = int(i % dim[k])
        i = (i - ret[k]) / dim[k]
    return ret


@jit
def cell_neighbours(ic: ndarray, ibox: ndarray) -> ndarray:
    ret: ndarray = np.zeros(3 ** ibox.shape[0], dtype=np.int64)
    for i, ind in enumerate(np.ndindex((3,) * ibox.shape[0])):
        # iterator of unraveled n-dimensional indices.
        ind = np.asarray(ind) - 1
        ret[i] = i_cell(ind + ic, ibox)
    return ret


@jit
def box_map(box: ndarray, r_cut: "ndarray or float") -> tuple:
    ibox: ndarray = np.asarray(box / r_cut, dtype=np.int64)
    ret: ndarray = np.zeros((np.multiply.reduce(ibox), 3 ** box.shape[0]))
    for ind in np.ndindex(ibox):
        ind = np.asarray(ind)
        ic = i_cell(ind, ibox)
        ret[ic] = cell_neighbours(ind, ibox)
    return ret, ibox


@jit
def linked_cl(pos: ndarray, box: ndarray, ibox: ndarray) -> tuple:
    head: ndarray = np.zeros(np.multiply.reduce(ibox), dtype=np.int64) - 1
    body: ndarray = np.zeros(pos.shape[0], dtype=np.int64) - 1
    counter: ndarray = np.zeros(head.shape, dtype=np.int64)
    for i in range(pos.shape[0]):
        ic = cell_id(pos[i], box, ibox)
        body[i] = head[ic]
        head[ic] = i
        counter[ic] += 1
    return head, body, counter


@jit
def get_from_cell(cid, head, body):
    ret = []
    j = head[cid]
    while j != -1:
        ret.append(j)
        j = body[j]
    return ret
