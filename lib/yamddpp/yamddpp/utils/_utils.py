from math import floor, sqrt

import numpy as np
from numba import cuda


@cuda.jit("float64(float64[:,:], float64[:], float64[:], float64[:])", device=True)
def cu_mat_dot_v_pbc_dist(a, b, c, box):
    ret = 0
    for i in range(a.shape[0]):
        tmp = 0
        for j in range(a.shape[1]):
            dc = b[j] - c[j]
            dc = dc - box[j] * floor(dc / box[j] + 0.5)
            tmp += a[i, j] * dc
        ret += tmp ** 2
    return sqrt(ret)


@cuda.jit("void(float64[:,:], float64[:], float64[:])", device=True)
def cu_mat_dot_v(a, b, ret):
    for i in range(a.shape[0]):
        tmp = 0
        for j in range(a.shape[1]):
            tmp += a[i, j] * b[j]
        ret[i] = tmp


@cuda.jit("void(float64[:,:], float64[:], float64[:], float64[:], float64[:])", device=True)
def cu_mat_dot_v_pbc(a, b, c, box, ret):
    for i in range(a.shape[0]):
        tmp = 0
        for j in range(a.shape[1]):
            dc = b[j] - c[j]
            dc = dc - box[j] * floor(dc / box[j] + 0.5)
            tmp += a[i, j] * dc
        ret[i] = tmp


@cuda.jit("float64(float64[:])", device=True)
def cu_v_mod(r):
    tmp = 0
    for i in range(r.shape[0]):
        tmp += r[i] ** 2
    return sqrt(tmp)


@cuda.jit('void(int64[:], int64)')
def cu_set_to_int(arr, val):
    i = cuda.grid(1)
    if i >= arr.shape[0]:
        return
    arr[i] = val


@cuda.jit('void(int64[:], int64[:])')
def cu_max_int(arr, ret):
    i = cuda.grid(1)
    if i >= arr.shape[0]:
        return
    cuda.atomic.max(ret, 0, arr[i])


@cuda.jit('float64(float64[:], float64[:], float64[:])', device=True)
def pbc_dist_cu(a, b, box):
    tmp = 0
    for i in range(a.shape[0]):
        d = b[i] - a[i]
        d = d - floor(d / box[i] + 0.5) * box[i]
        tmp += d * d
    return sqrt(tmp)


@cuda.jit("void(int64, int64[:], int64[:])", device=True)
def unravel_index_f_cu(i, dim, ret):  # unravel index in Fortran way.
    for k in range(dim.shape[0]):
        ret[k] = int(i % dim[k])
        i = (i - ret[k]) / dim[k]


@cuda.jit("int64(int64[:], int64[:])", device=True)
def ravel_index_f_cu(i, dim):  # ravel index in Fortran way.
    ret = (i[0] + dim[0]) % dim[0]
    tmp = dim[0]
    for k in range(1, dim.shape[0]):
        ret += ((i[k] + dim[k]) % dim[k]) * tmp
        tmp *= dim[k]
    return ret


@cuda.jit("void(int64[:], int64[:])", device=True)
def add_local_arr_mois_1(a, b):
    for i in range(a.shape[0]):
        a[i] = a[i] + b[i] - 1


def rfft2fft(rfft, n):
    r"""Extend rfft output to FFT output.

    See numpy.fft.rfftn for details, the last axis of output of rfftn is always
    n // 2 + 1.

    Example:

    >>> a = np.random.random((10,20,30))
    >>> np.allclose(rfft2fft(np.fft.rfftn(a), 30), np.fft.fftn(a))
    True

    :param rfft: np.ndarray
    :param n: int, last axis of desired fft results
    :return: np.ndarray, fft results.
    :raises: ValueError if n // 2 + 1 is not rfft.shape[-1]
    """
    if n // 2 + 1 != rfft.shape[-1]:
        raise ValueError("The sizes of last axis of fftn (n_fftn) outputs and rfftn"
                         " outputs (n_rfftn) must satisfy n_fftn // 2 + 1 == n_rfftn!")
    n_dim = rfft.ndim
    fslice = tuple([slice(0, _) for _ in rfft.shape])
    lslice = np.arange(n - n // 2 - 1, 0, -1)
    pad_axes = [(0, 1)] * (n_dim - 1) + [(0, 0)]
    flip_axes = tuple(range(n_dim - 1))
    # fftn(a) = np.concatenate([rfftn(a),
    # conj(rfftn(a))[-np.arange(i),-np.arange(j)...,np.arange(k-k//2-1,0,-1)]], axis=-1)
    return np.concatenate([rfft, np.flip(np.pad(rfft.conj(), pad_axes, 'wrap'),
                                         axis=flip_axes)[fslice][..., lslice]], axis=-1)
