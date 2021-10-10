import numpy as np
from ..utils import pbc

def rdf_xy(x, y, box, dr=None, bins=None, use_gpu=False):
    """Calcualte radial distribution function
    :param x        : np.ndarray, positions
    :param y        : np.ndarray, positions
    :param dr       : float, gr spacing dr
    :param bins     : int, number of dr bins 
    :param use_gpu  : bool, calculate with gpu 

    todo: (1) bond/angle/mol exlcusions (2) fft (3) gpu (4) cufft
    """
    if dr   is None: dr   = 0.05
    if bins is None: bins = int(box.max() / 2 / dr)
    mode = 'xx' if x is y else 'xy' 

    rmax = dr * bins
    rho = x.shape[0] / box[0] / box[1] / box[2]
    if use_gpu is True:
        pass # todo
    else:
        rij = np.linalg.norm(pbc(x[:, None, :] - y[None, :, :], box), axis=-1)

    if mode == 'xx':
        rij = rij[np.triu_indices_from(rij, k=1)]
    hist, edges = np.histogram(rij, bins=bins, range=(0, rmax))
    vol = 4 * np.pi / 3.0 * (edges[1:]**3 - edges[:-1]**3)
    gr = mode.count('x') * hist / rho / vol / x.shape[0]
    r_mid = edges[1:] - dr / 2

    return np.c_[r_mid, gr]

if __name__ == '__main__':
    pass
