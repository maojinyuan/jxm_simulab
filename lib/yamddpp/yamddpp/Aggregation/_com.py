import numpy as np

_bins = 500
_bin_size = 2 * np.pi / _bins
_d = np.linspace(0, 2 * np.pi, _bins, endpoint=False)


def _circfuncs_common(samples, high, low):
    r"""
    :param samples: np.ndarray
    :param high: np.ndarray
    :param low: np.ndarray
    :return: np.ndarray
    """
    samples = np.asarray(samples)
    if samples.size == 0:
        return np.nan
    return (samples - low) * 2. * np.pi / (high - low)


def _circ_mean(ang, axis=None, mass=1):
    r"""Circular mean of angles in [0, 2\pi).
    :param ang: np.ndarray, angular values in [0, 2\pi).
    :param axis: int, about which axis
    :param mass: np.ndarray or float, mass of particles.
    :return: np.ndarray
    """
    s = (mass * np.sin(ang)).sum(axis=axis)
    c = (mass * np.cos(ang)).sum(axis=axis)
    res = np.arctan2(-s, -c)
    return res + np.pi


def _circ_midpoint(x, _bin_size, _bins):
    r"""Deduplicate using histogram then calculate circular median.
    :param x: np.ndarray.
    :param _bin_size: float, bin size.
    :param _bins: int, how many bins.
    :return: np.ndarray
    """
    p = np.bincount((x / _bin_size).astype(np.int), minlength=_bins)
    return _circ_mean(_d[p > 0])


def com(x, hi, lo, axis=0, mass=1, midpoint=False):
    r"""Compute COM or median of PBC datas.
    :param x: np.ndarray, input coordinates.
    :param hi: np.ndarray, box.high
    :param lo: np.ndarray, box.low
    :param axis: int, mean along given axis.
    :param mass: np.ndarray or float, mass of particles.
    :param midpoint: bool, calculate midpoint or COM
    :return: np.ndarray, midpoint or COM
    """
    hi = np.asarray(hi)
    lo = np.asarray(lo)
    box = hi - lo
    x_ang = _circfuncs_common(x, hi, lo)
    if not midpoint:
        mean_ang = _circ_mean(x_ang, axis=axis, mass=mass)
    else:
        mean_ang = np.apply_along_axis(_circ_midpoint, axis,
                                       x_ang, _bin_size, _bins)
    return mean_ang / 2 / np.pi * box + lo
