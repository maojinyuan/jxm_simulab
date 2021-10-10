import numpy as np

from ._mat_ac import mat_ac
from ._mat_ac import vec_ac


def cross_correlate(in1, in2, axis=0):
    r"""Cross-correlation of matrices along given axis.
    Usually used in time-correlation, one axis is enough. Don't
    use this function to calculate autocorrelation, for
    cross_correlate(in1, in1[::-1]) is on interval [-len(in1), len(in1)),
    not on [0, len(in1)), in this case, output[len(in1)-1:] is the correct
    output for in1 == in2 or in1.size == in2.size, due to `np.flip` is
    applied on in2 here. Or generally,
    ifft(fft(a, n=na+nb-1) * fft(b, n=na+nb-1).conj())[:na] ==
    np.correlate(a, b, 'full')[nb-1:] for na >= nb... See `same' and `valid' mode
    in `scipy.fft.convolve' (`_centered' function).

    >>> a = np.random.random(10)
    >>> b = np.random.random(10)
    >>> np.allclose(ifft(fft(a,19)*fft(b,19).conj())[:10], np.correlate(a,b,'full')[9:])
    True

    :param in1: np.ndarray
    :param in2: np.ndarray
    :param axis: int
    :return: np.ndarray, `full' mode of np.correlate result.
    """
    fft = np.fft.rfft
    ifft = np.fft.irfft
    if (np.issubdtype(in1.dtype, np.complex) or
            np.issubdtype(in2.dtype, np.complex)):
        fft = np.fft.fft
        ifft = np.fft.ifft
    s1 = in1.shape[axis]
    s2 = in2.shape[axis]
    s = s1 + s2 - 1
    return ifft(fft(in1, axis=axis, n=s) *
                fft(np.flip(in2, axis=axis).conj(), axis=axis, n=s),
                axis=axis, n=s)[:s]
