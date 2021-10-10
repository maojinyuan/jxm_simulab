import numpy as np


def _vec_cc(a, b, n):
    r"""$FT{a \star b + b \star a}$
    :param a: np.ndarray, a
    :param b: np.ndarray, b
    :return: np.ndarray
    """
    return (np.fft.rfft(a, axis=0, n=n).conj() * np.fft.rfft(b, axis=0, n=n)).real * 2


def mqd(x, cum=True):
    r"""<|r(t)-r(0)|^4>.
    :param x: np.ndarray, (n_frames, n_particles, n_dim)
    :param cum: bool, summing n_particles or not.
    :return: np.ndarray, mqd
    """
    n, n_samples = x.shape[0], x.shape[1]
    s = n * 2
    x2 = np.square(x)
    x4 = np.square(x2).sum(axis=2)
    xt, yt, zt = x[..., 0], x[..., 1], x[..., 2]
    xt2, yt2, zt2 = x2[..., 0], x2[..., 1], x2[..., 2]
    x4 += 2 * (xt2 * yt2 + yt2 * zt2 + zt2 * xt2)
    if cum:
        x4 = x4.sum(axis=1)
    x0 = 2 * x4.sum(axis=0)
    _shape = (n,) if cum else (n, n_samples)
    xm = np.zeros(_shape)
    xm[0] = x0
    for m in range(1, n):
        x0 = x0 - x4[m - 1] - x4[n - m]
        xm[m] = x0
    norm = np.arange(n, 0, -1)
    am = np.zeros((s // 2 + 1, n_samples))  # length of s-length rfft output.
    am += 6 * np.abs(np.fft.rfft(xt2, axis=0, n=s)) ** 2
    am += 6 * np.abs(np.fft.rfft(yt2, axis=0, n=s)) ** 2
    am += 6 * np.abs(np.fft.rfft(zt2, axis=0, n=s)) ** 2
    am += 8 * np.abs(np.fft.rfft(xt * yt, axis=0, n=s)) ** 2
    am += 8 * np.abs(np.fft.rfft(xt * zt, axis=0, n=s)) ** 2
    am += 8 * np.abs(np.fft.rfft(yt * zt, axis=0, n=s)) ** 2
    am += 2 * _vec_cc(xt2, yt2, s)
    am += 2 * _vec_cc(xt2, zt2, s)
    am += 2 * _vec_cc(yt2, zt2, s)
    _tmp = np.fft.rfft(xt, n=s, axis=0)
    am += -4 * (np.fft.rfft(xt ** 3, axis=0, n=s).conj() * _tmp).real * 2
    am += -4 * (np.fft.rfft(xt * yt2, axis=0, n=s).conj() * _tmp).real * 2
    am += -4 * (np.fft.rfft(xt * zt2, axis=0, n=s).conj() * _tmp).real * 2
    _tmp = np.fft.rfft(yt, n=s, axis=0)
    am += -4 * (np.fft.rfft(yt ** 3, axis=0, n=s).conj() * _tmp).real * 2
    am += -4 * (np.fft.rfft(xt2 * yt, axis=0, n=s).conj() * _tmp).real * 2
    am += -4 * (np.fft.rfft(yt * zt2, axis=0, n=s).conj() * _tmp).real * 2
    _tmp = np.fft.rfft(zt, n=s, axis=0)
    am += -4 * (np.fft.rfft(zt ** 3, axis=0, n=s).conj() * _tmp).real * 2
    am += -4 * (np.fft.rfft(xt2 * zt, axis=0, n=s).conj() * _tmp).real * 2
    am += -4 * (np.fft.rfft(yt2 * zt, axis=0, n=s).conj() * _tmp).real * 2
    if cum:
        am = am.sum(axis=1)
    else:
        norm = np.expand_dims(norm, axis=-1)
    return (xm + np.fft.irfft(am, axis=0, n=s)[:n]) / norm
