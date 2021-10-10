import numpy as np


def vec_ac(x, cum=None):
    r"""Vector autocorrelation function with samples.
    :param x: np.ndarray -> (n_frames, n_vectors, ..., ndim), ndim == 1 for 1-d vectors
    :param cum: int or seq of ints, take average of n_vectors or not.
    :return: np.ndarray -> (n_frames,) of vector autocorrelation
            or (n_frames, n_vectors) if `cum=False'.
    """
    fft = np.fft.rfft
    ifft = np.fft.irfft
    if np.issubdtype(x.dtype, np.dtype(np.complex)):
        fft = np.fft.fft
        ifft = np.fft.ifft
    n = x.shape[0]
    s = 2 * n
    if cum is not None:
        if isinstance(cum, int): cum = (cum,)
        if 0 in cum: raise ValueError("Time axis cannot be summed!")
        summing_axes = tuple((*cum, -1)) if -1 not in cum else tuple(cum)
    else:
        summing_axes = (-1,)  # only add dimension
    # summing over samples and dimension or just dimension
    norm = np.arange(n, 0, -1)
    norm = np.expand_dims(norm, axis=tuple(range(1, x.ndim - len(summing_axes))))
    # the dimension of correlation is at most (x.ndim - 1) for the last axis is
    # summed due to vector dot. Or, if the cum axes are given, the dimension of
    # output is (x.ndim - len(summing_axes)), an easier solution is setting
    # `keepdims=True` for `np.sum`, then the dimension of array is preserved.
    # am I so stupid that I've forgotten the reason why I wrote the mat_ac function???
    return ifft(np.sum(abs(fft(x, axis=0, n=s)) ** 2,
                       axis=summing_axes), axis=0, n=s)[:n].real / norm


def mat_ac(x, axes=None):
    r"""Matrix autocorrelation function.
    :param x: np.ndarray -> (n_frames, ...) of input
    :param axes: tuple or int, axes that summing up.
    :return: np.ndarray -> (n_frames, ...) of output
    :raises: ValueError, if axes contains 0.
    """
    fft = np.fft.rfft
    ifft = np.fft.irfft
    if np.issubdtype(x.dtype, np.dtype(np.complex)):
        fft = np.fft.fft
        ifft = np.fft.ifft
    n = x.shape[0]
    s = 2 * n  # 2 * n - 1 is fine.
    norm = np.arange(n, 0, -1).reshape(n, *[1] * (x.ndim - 1))
    if axes is None:
        return ifft(abs(fft(x, axis=0, n=s)) ** 2,
                    axis=0, n=s)[:n].real / norm
    else:
        axes = np.atleast_1d(np.asarray(axes, dtype=np.int))
        if 0 in axes:
            raise ValueError("The 1st axis should be time axis!")
        norm = norm.reshape(n, *[1] * (x.ndim - 1 - axes.size))
        return ifft(np.sum(abs(fft(x, axis=0, n=s)) ** 2, axis=tuple(axes)),
                    axis=0, n=s)[:n].real / norm
