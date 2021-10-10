cimport cython
import numpy as np

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cell_id(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] box, np.ndarray[long, ndim=1] ibox):
    cdef np.ndarray[long, ndim=1] ind = np.asarray(((x / box + 0.5) * ibox), dtype=np.int64)
    return ind[0] + ind[1] * ibox[0] + ind[2] * ibox[1] * ibox[0]

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def i_cell(np.ndarray[long, ndim=1] cid, np.ndarray[long, ndim=1] ibox):
    cdef np.ndarray[long, ndim=1] ind = np.asarray((cid + ibox) % ibox, dtype=np.int64)
    return ind[0] + ind[1] * ibox[0] + ind[2] * ibox[1] * ibox[0]

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def unravel_index_f(long i, np.ndarray[long, ndim=1] dim):
    cdef long k
    cdef np.ndarray[long, ndim=1] ret = np.zeros(dim.shape, dtype=np.int64)
    for k in range(dim.shape[0]):
        ret[k] = int(i % dim[k])
        i = (i - ret[k]) / dim[k]
    return ret

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cell_neighbours(np.ndarray[long, ndim=1] ic, np.ndarray[long, ndim=1] ibox):
    cdef long i
    cdef long n = 3 ** ibox.shape[0]
    cdef np.ndarray[long, ndim=1] shape = np.zeros(ibox.shape, dtype=np.int64) + 3
    cdef np.ndarray[long, ndim=1] ret = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ind = unravel_index_f(i, shape) - 1
        ret[i] = i_cell(ind + ic, ibox)
    return ret

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def box_map(np.ndarray[double, ndim=1] box, double r_cut):
    cdef np.ndarray[long, ndim=1] ibox = np.asarray(box / r_cut, dtype=np.int64)
    cdef long ic, j
    cdef long n = np.multiply.reduce(ibox)
    cdef np.ndarray[long, ndim=2] ret = np.zeros((n, ibox.shape[0] ** 3))
    for j in range(n):
        ind = unravel_index_f(j, ibox)
        ic = i_cell(ind, ibox)
        ret[ic] = cell_neighbours(ind, ibox)
    return ret, ibox

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def linked_cl(np.ndarray[double, ndim=2] pos, np.ndarray[double, ndim=1] box,
              np.ndarray[long, ndim=1] ibox):
    cdef np.ndarray[long, ndim=1] head = np.zeros(np.multiply.reduce(ibox), dtype=np.int64) - 1
    cdef np.ndarray[long, ndim=1] body = np.zeros(pos.shape[0], dtype=np.int64)
    cdef long i
    for i in range(pos.shape[0]):
        ic = cell_id(pos[i], box, ibox)
        body[i] = head[ic]
        head[ic] = i
    return head, body
