from cmath import exp as cexp
from math import sqrt, pi, fmod, gamma

import numpy as np
from numba import cuda, int32, float32, float64, complex64, complex128


def gen_sph(dtype):
    if dtype == np.dtype(np.float64):
        float = float64
        complex = complex128
    else:
        float = float32
        complex = complex64

    @cuda.jit(float(int32, int32, float), device=True)
    def legendre(l, m, x):
        pmm = 1.0
        if m > 0:
            somx2 = sqrt((1. - x) * (1. + x))
            fact = 1.0
            for i in range(1, m + 1):
                pmm *= -fact * somx2
                fact += 2.0
        if l == m:
            return pmm
        pmmp1 = x * (2 * m + 1) * pmm
        if l == m + 1:
            return pmmp1
        for ll in range(m + 2, l + 1):
            pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
            pmm = pmmp1
            pmmp1 = pll
        return pll

    @cuda.jit(complex(int32, int32, float, float), device=True)
    def sphHar(l, m, cosTheta, phi):
        m1 = abs(m)
        c = sqrt((2 * l + 1) * gamma(l - m1 + 1.) / (4 * pi * gamma(l + m1 + 1.)))
        c *= legendre(l, m1, cosTheta)
        y = cexp(m * phi * 1j)
        if fmod(m, 2) == -1.:
            y *= -1
        return y * c + 0j

    return sphHar
