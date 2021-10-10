import numpy as np

from ..utils import hist_vec_by_r
from ..utils import hist_vec_by_r_cu


def rdf_xy(x, y, x_range, bins, r_bin=0.2, use_gpu=False):
    r"""Calculate RDF by FFT.
    :param x: np.ndarray, coordinates of component 1
    :param y: np.ndarray, coordinates of component 2
    :param x_range: np.ndarray, range of positions
    :param bins: np.ndarray or int, bins in calculation of density matrix
    :param r_bin: double, bin size of r
    :param use_gpu: bool or int, use gpu code to summing vector g(R) to g(r)
    :return: tuple of r and g(r)
    """
    if not (isinstance(use_gpu, bool) or isinstance(use_gpu, int)):
        raise ValueError("`use_gpu' should be bool or int!")
    mode = 'ab' if x is not y else 'aa'
    bins = np.asarray(bins)
    bins = np.where(bins % 2 == 0, bins + 1, bins)
    # better use odd number here. different from sq, where dq has 2\pi
    # in it, here we use, which might be 0.1, 0.01, etc. For instance,
    # let r_bin=0.1, r_cut=0.5, and using x = np.random.random((n, 3))
    # to generate samples, set box to be (0, 1] and bins = (6,6,6), may return
    # different results if one sets box to be (-0.5, 0.5] and x = x - 0.5;
    # some numerical errors cause a wired phenomenon: in case of FT{rho(x)}[5,5,4],
    # with shift [3,3,3], the vector is [2,2,1] / 6 which has modulus of exact 0.5,
    # and the index is int(0.5/0.1) = 5; however, if we shift box to (-0.5, 0.5),
    # the modulus of this vector turns out to be 0.499999... and index is 4. All other
    # parameters are same, both cuda ver and cpu ver hist_vec_by_r function give
    # different value. This strange bug vanishes sometimes with even number of bins,
    # but never occurs in case of odd number of bins.
    # results from hist_vec_to_r:
    """
    In[7]: hist_vec_by_r(a, 1 / 6, 0.1, 0.5, middle=np.array([3, 3, 3.]))
    Out[7]: array([1., 6., 20., 30., 36., 1.])  # the counter, a.shape = (6,6,6)
    In [8]: hist_vec_by_r(a, 0.1666666666666, 0.1, 0.5, middle=np.array([3,3,3.]))                                                                 
    Out[8]: array([ 1.,  6., 20., 30., 63.,  1.])
    but in this rdf case, variables are x_range=[(0.0,1.0)]*3 or [(-0.5, 0.5)]*3
    and x = np.random.random((n, 3)) and x = x - 0.5, in these 2 cases, even the
    _rdf_xyz before hist_vec_to_r are same. I have double checked r_cut, r_bin, dr, etc.
    the ***PRINT*** results are also same. I don't know what triggers this. Now the solution
    is simply use odd numbers as bin dimensions.
    """
    box = np.array(np.array([_[1] - _[0] for _ in x_range]))
    px, ex = np.histogramdd(x, bins=bins, range=x_range)
    _ft_px = np.fft.rfftn(px)
    _ft_py = _ft_px
    if mode == 'ab':
        py, ex = np.histogramdd(y, bins=bins, range=x_range)
        _ft_py = np.fft.rfftn(py)
    _ft_px_py = _ft_px * _ft_py.conj()
    _rdf_xyz = np.fft.irfftn(_ft_px_py, bins)
    # _ft_py_px[t] == _ft_px_py[-t]
    _rdf_xyz[0, 0, 0] -= 0 if mode == 'ab' else x.shape[0]
    _rdf_xyz = np.fft.fftshift(_rdf_xyz)  # for x, y are in (-box/2, box/2]
    # if rdf is not shifted, it becomes [0, dr, 2dr, ..., n//2 dr, -n//2 dr, ..., -dr]
    middle = np.asarray(_rdf_xyz.shape, dtype=np.float64) // 2
    dr = ex[0][1] - ex[0][0]
    if use_gpu is False:
        _rdf = hist_vec_by_r(_rdf_xyz, dr, r_bin, box.min() / 2, middle=middle)
    else:
        _rdf = hist_vec_by_r_cu(_rdf_xyz, dr, r_bin, box.min() / 2,
                                gpu=use_gpu, middle=middle)
    _rdf /= x.shape[0] * y.shape[0]
    _rdf *= np.multiply.reduce(bins)
    return (np.arange(_rdf.shape[0]) + 0.5) * r_bin, _rdf
