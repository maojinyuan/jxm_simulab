import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange, parallel
from libc.math cimport floor,sqrt,pow
cimport
numpy as np
import cython
import numpy as np
from cython.parallel import prange, parallel
from libc.math cimport

floor, sqrt, pow


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef double pbc_dist(double[:] x, double[:] y, double[:] b) nogil:
    cdef int i, d
    cdef double tmp=0, r=0
    d = b.shape[0]
    for i in range(d):
        tmp = x[i]-y[i]
        tmp = tmp - b[i] * floor(tmp/b[i]+0.5)
        r = r + pow(tmp, 2)
    return sqrt(r)


@cython.wraparound(False)
@cython.boundscheck(False)
def pdist_omp(double[:,:] x, double[:] box):
    cdef int i, j, k, l, s, n, d
    cdef np.ndarray[np.double_t, ndim=1] ret
    n = x.shape[0]
    d = x.shape[1]
    s = n * (n - 1) / 2
    ret = np.zeros(s, dtype=np.float)
    with nogil, parallel():
        for i in prange(n-1, schedule='static'):
            for j in range(i + 1, n):
                l = i * (n - 1) - i * (i + 1) / 2 + j - 1
                ret[l] = pbc_dist(x[i], x[j], box)
    return ret


@cython.wraparound(False)
@cython.boundscheck(False)
def cdist_omp(double[:,:] x, double[:,:] y, double[:] box):
    cdef int i, j,  m, n
    cdef np.ndarray[np.double_t, ndim=2] ret
    m = x.shape[0]
    n = y.shape[0]
    d = x.shape[1]
    ret = np.zeros((m, n), dtype=np.float)
    with nogil, parallel():
        for i in prange(m, schedule='static'):
            for j in range(n):
                ret[i, j] = pbc_dist(x[i], y[j], box)
    return ret
