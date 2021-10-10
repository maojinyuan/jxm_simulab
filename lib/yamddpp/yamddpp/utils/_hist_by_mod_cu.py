from math import floor, sqrt

import numba as nb
import numpy as np
from numba import cuda, int64


@cuda.jit("void(int64, int64[:], int64[:])", device=True)
def unravel_index_f_cu(i, dim, ret):  # unravel index in Fortran way.
    for k in range(dim.shape[0]):
        ret[k] = int(i % dim[k])
        i = (i - ret[k]) / dim[k]


def _unravel_indices_f(dim, gpu=0):
    n_dim = dim.shape[0]
    n = np.multiply.reduce(dim)
    ret = np.zeros((n, n_dim), dtype=np.int64)
    dim = np.asarray(dim, dtype=np.int64)

    @cuda.jit("void(int64[:], int64, int64[:,:])")
    def unravel_indices_f_cu(_dim, _n, _ret):
        i = cuda.grid(1)
        tmp = cuda.local.array(n_dim, nb.int64)  # n_dim must be constant in this def.
        # private array for every thread.
        if i < n:
            unravel_index_f_cu(i, _dim, tmp)
            for j in range(tmp.shape[0]):
                _ret[i, j] = tmp[j]

    with cuda.gpus[gpu]:
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        bpg = int(np.ceil(n / tpb))
        unravel_indices_f_cu[bpg, tpb](dim, n, ret)
    return ret


@cuda.jit("void(float64[:], int64[:], float64[:], float64, float64, float64, float64[:], int64[:])")
def _cu_kernel(x, dim, middle, dr, r_bin, r_max2, ret, cter):
    i = cuda.grid(1)
    if i >= x.shape[0]:
        return
    tmp = 0
    j = i
    for k in range(dim.shape[0]):  # r.shape == (n-dim, n-coordinates in each dim)
        idx = int(j % dim[k])
        j = (j - idx) / dim[k]
        tmp += (dr * (idx - middle[k])) ** 2
    if tmp < r_max2:
        jdx = int(tmp ** 0.5 / r_bin)
        cuda.atomic.add(ret, jdx, x[i])
        cuda.atomic.add(cter, jdx, 1)


# raveled array, in fortran way !!!
@cuda.jit("void(float64[:], float64[:], int64[:], float64[:], float64,"
          "float64, int64, float64[:], float64[:], int64[:])")
def _cu_kernel_complex(x_real, x_imag, dim, middle, dr, r_bin, r_max2, ret, ret_imag, cter):
    i = cuda.grid(1)
    if i >= x_real.shape[0]:
        return
    tmp = 0
    j = i
    for k in range(dim.shape[0]):  # r.shape == (n-dim, n-elements on each dim)
        idx = int(j % dim[k])
        j = (j - idx) / dim[k]
        tmp += (dr * (idx - middle[k])) ** 2
        # unraveled in Fortran way !!!
    if tmp < r_max2:
        jdx = int(tmp ** 0.5 / r_bin)
        # jdx = int(tmp ** 0.5 / r_bin) # this method is not accurate for r_max = r_bin * nbins
        # it should be r_max = (int(r_max / r_bin) + 1) * r_bin
        # jdx = int64(floor(sqrt(tmp) / r_bin))  # also not that accurate, so use index in the if sentence
        cuda.atomic.add(ret, jdx, x_real[i])  # currently cuda.atomic.add does not support np.complex
        cuda.atomic.add(ret_imag, jdx, x_imag[i])
        cuda.atomic.add(cter, jdx, 1)


def hist_vec_by_r_cu(x, dr, r_bin, r_max, middle=None, gpu=0):
    r"""Summing vector based function to modulus based function.
    $f(r) := \int F(\bm{r})\delta(r-|\bm{r}|)\mathrm{d}\bm{r} / \int \delta(r-|\bm{r}|)\mathrm{d}\bm{r}$
    :param x: np.ndarray, input
    :param r: np.ndarray[ndim=2], x[len_1, len_2, ..., len_n] ~ (r1: len_1, r2: len_2, ..., r_n: len_n)
    x: (Nx, Ny, Nz) -> r: (3 (xyz), N), currently, only Nx == Ny == Nz is supported.
    :param r_bin: double, bin size of r
    :param r_max: double, max of r
    :param gpu: int gpu number
    :return: np.ndarray, averaged $F(x, y, ...) -> 1/(4\pi\r^2) f(\sqrt{x^2+y^2+...})$
    """
    dim = np.asarray(x.shape, dtype=np.int64)
    ret = np.zeros(int(r_max / r_bin) + 1, dtype=np.float)
    r_max2 = float((ret.shape[0] * r_bin) ** 2)
    cter = np.zeros(ret.shape, dtype=np.int64)
    x = x.ravel(order='F')
    if middle is None:
        middle = np.zeros(dim.shape[0], dtype=np.float64)
    with cuda.gpus[gpu]:
        device = cuda.get_current_device()
        tpb = device.WARP_SIZE
        bpg = int(x.shape[0] // tpb + 1)
        if np.issubdtype(x.dtype, np.dtype(np.complex)):
            x_real = np.ascontiguousarray(x.real)
            x_imag = np.ascontiguousarray(x.imag)
            ret_imag = np.zeros(int(r_max / r_bin) + 1, dtype=np.float)
            _cu_kernel_complex[bpg, tpb](x_real, x_imag, dim, middle, dr, r_bin, r_max2, ret, ret_imag, cter)
            ret = ret + ret_imag * 1j
        else:
            _cu_kernel[bpg, tpb](x, dim, middle, dr, r_bin, r_max2, ret, cter)
    cter[cter == 0] = 1
    return ret / cter
