import numpy as np

from ..utils import hist_vec_by_r
from ..utils import hist_vec_by_r_cu


def scatter_xy(x, y=None, x_range=None, r_cut=0.5, q_bin=0.1, q_max=6.3, zero_padding=1, expand=0, use_gpu=False):
    r"""Calculate static structure factor.
    :param x: np.ndarray, coordinates of component 1
    :param y: np.ndarray, coordinates of component 2
    :param x_range: np.ndarray, range of positions
    :param r_cut: double, bin size of rho
    :param q_bin: double, bin size of wave vector q
    :param q_max: double, max value of wave vector q
    :param zero_padding: int (periods), whether pad density matrix with 0
    :param expand: int or np.ndarray (periods), extend density matrix by its period
    :param use_gpu: bool or int, use gpu code to summing vector S(Q) to S(q)
    :return: np.ndarray, S(q) value
    """
    mode = 'ab' if x is not y else 'aa'
    # Using `x_range' rather than `box' for the unknown origin of the box
    box = np.array(np.array([_[1] - _[0] for _ in x_range]))
    bins = np.asarray(box / r_cut, dtype=np.int)
    x_range = np.asarray(x_range)
    expand = np.asarray(expand)
    n_dim = x.shape[1]
    if x_range.shape[0] != n_dim:
        raise ValueError("Dimension of coordinates is %d and"
                         "dimension of x_range is %d" % (n_dim, x_range.shape[0]))
    if bins.ndim < 1:
        bins = np.asarray([bins] * n_dim)
    if not (isinstance(use_gpu, bool) or isinstance(use_gpu, int)):
        raise ValueError(
            "`use_gpu' should be bool: False for not using GPU or an integer of GPU id!"
        )
    rho_x, _ = np.histogramdd(x, bins=bins, range=x_range)
    if expand.ndim < 1:
        expand = np.asarray([expand] * rho_x.ndim)
    z_bins = (np.asarray(rho_x.shape) * zero_padding).astype(np.int64)
    rho_x = np.pad(rho_x, [(0, _ * __) for _, __ in zip(rho_x.shape, expand)], 'wrap')
    z_bins = np.where(
        z_bins > np.asarray(rho_x.shape[0]), z_bins, np.asarray(rho_x.shape[0])
    )
    _rft_sq_x = np.fft.rfftn(rho_x, s=z_bins)
    # expand density with periodic data, enlarge sample periods.
    _rft_sq_y = _rft_sq_x
    if mode == 'ab':
        rho_y, _ = np.histogramdd(y, bins=bins, range=x_range)
        rho_y = np.pad(rho_y, [(0, _ * __) for _, __ in zip(rho_y.shape, expand)], 'wrap')
        _rft_sq_y = np.fft.rfftn(rho_y, s=z_bins)
    _rft_sq_xy = _rft_sq_x.conj() * _rft_sq_y  # circular correlation.
    fslice = tuple([slice(0, _) for _ in z_bins])
    lslice = np.arange(z_bins[-1] - z_bins[-1] // 2 - 1, 0, -1)
    pad_axes = [(0, 1)] * (n_dim - 1) + [(0, 0)]
    flip_axes = tuple(range(n_dim - 1))
    # fftn(a) = np.concatenate([rfftn(a),
    # conj(rfftn(a))[-np.arange(i),-np.arange(j)...,np.arange(k-k//2-1,0,-1)]], axis=-1)
    # numpy >= 1.15
    # The pad is to ensure arr -> arr[0,-1,-2,...] (arr[0, N-1...1] not flip(arr)->arr[-1,-2,...]
    # (arr[N-1,N-2,...0]
    _sq_xy = np.concatenate(
        [_rft_sq_xy, np.flip(
            np.pad(_rft_sq_xy.conj(), pad_axes, 'wrap'), axis=flip_axes
        )[fslice][..., lslice]], axis=-1
    )
    # np.fft.rfftfreq does not work here, it has be the complete fft result.
    _d = box / bins
    # q = np.vstack([np.fft.fftfreq(_sq_xy.shape[_], _d[_]) for _ in range(_d.shape[0])])
    q0 = np.fft.fftfreq(_sq_xy.shape[0], _d[0])
    # _d is same in all directions, i.e., r_cut of sampling is same in all directions
    # so that dq is same in all directions
    dq = q0[1] - q0[0]
    dq = dq * 2 * np.pi
    middle = np.asarray(_sq_xy.shape, dtype=np.float64) // 2
    _sq_xy = np.fft.fftshift(_sq_xy)  # shift 0-freq to middle
    # _sq_xy[0, 0, ..., 0] = np.fft.fftshift(_sq_xy)[middle]
    if use_gpu is False:
        return hist_vec_by_r(_sq_xy, dq, q_bin, q_max, middle=middle)
    return hist_vec_by_r_cu(_sq_xy, dq, q_bin, q_max, gpu=use_gpu, middle=middle)
