from math import floor, sqrt

import numpy as np
from numba import cuda
from numba import float64
from numba import vectorize

from ..utils import get_from_cell
from ..utils import linked_cl
from ._cluster_dbscan import cluster as method_dbscan
from ._com import com


# TODO: different cluster methods.


@vectorize([(float64(float64, float64))], target='parallel')  # gpu, cpu
def pbc_ufunc(r, d):
    return r - d * floor(r / d + 0.5)


# faster than pbc defined below


def pbc(p, d):
    return p - d * np.round(p / d)


def handle_clusters(clusters, pos, types, box, bins=50):
    r"""Handle the results of clustering.
    :param clusters: list, list of clusters/
    :param pos: np.ndarray, positions
    :param types: np.ndarray, types
    :param box: np.ndarray, box
    :param bins: int, bins to check percolation.
    :return: None
    """
    meta = open('cluster_meta.txt', 'w')
    fmt = '%04d %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n'
    for i, cluster in enumerate(clusters):
        p_cluster = pos[cluster]
        p_types = types[cluster]
        percolate = np.asarray(
            [np.all(np.histogram(_, bins=bins, range=(-__ / 2, __ / 2))[0] > 0)
             for _, __ in zip(p_cluster.T, box)], dtype=np.bool
        )
        # center_of_mass = com(p_cluster, box / 2, -box / 2)
        # midpoint = com(p_cluster, box / 2, -box / 2, midpoint=True)
        # midpoint[percolate] = center_of_mass[percolate]  # using midpoint if not percolate, else center of mass
        percolate = np.logical_not(percolate)  # midpoint=True if percolate == False
        midpoint = np.asarray(
            [com(coors, l / 2, -l / 2, midpoint=percolate_p) for coors, l, percolate_p
             in zip(p_cluster.T, box, percolate)]
        )  # using midpoint=True if not percolate, else com
        # calculate in each dimension, faster.
        p_cluster = pbc(p_cluster - midpoint, box)  # always in (midpoint-box/2, midpoint+box/2)
        p_cluster -= p_cluster.mean(axis=0)  # make com be 0
        rg_tensor = p_cluster.T.dot(p_cluster) / p_cluster.shape[0]
        e, v = np.linalg.eig(rg_tensor)
        meta.write(fmt % (i, p_cluster.shape[0], e[0], e[1], e[2],
                          v.T[0, 0], v.T[0, 1], v.T[0, 2],
                          v.T[1, 0], v.T[1, 1], v.T[1, 2],
                          v.T[2, 0], v.T[2, 1], v.T[2, 2]))
        xyz = open('%04d.xyz' % i, 'w')
        xyz.write('%d\nmeta\n' % (p_cluster.shape[0]))
        for __, _ in zip(p_types, p_cluster):
            xyz.write('%s %.4f %.4f %.4f\n' % (__, _[0], _[1], _[2]))
        xyz.close()
        meta.close()


def coarse_grained_cluster(pos, box, method, kwargs=None, r_cut=0):
    r"""Cluster particles directly or cluster the cells to reduce calculation.

    :param pos: np.ndarray, positions
    :param box: np.ndarray, box
    :param method: callable, clustering method
    :param kwargs: dictionary, args of func
    :param r_cut: float or np.ndarray, coarse-grain size, 0 for clustering directly.
    :return: list, clusters with particles ids.
    """
    if kwargs is None:
        kwargs = {}
    if not r_cut:
        return method(pos, box, **kwargs)
    bins = np.asarray(box / r_cut, dtype=np.int)
    # weights, _ = np.histogramdd(pos, bins=bins, range=[(-_ / 2, _ / 2) for _ in box])
    # weights = weights.ravel(order='F')  # ravel in Fortran
    # build the cell-list at the same time
    head, body, weights = linked_cl(pos, box, bins)  # weights was already raveled in Fortran way
    coordinates = np.vstack(np.unravel_index(np.arange(weights.shape[0]), bins, order='F')).T
    coordinates = coordinates * r_cut
    fitted = method(coordinates, box, **kwargs)
    clusters = [np.arange(head.shape[0])[fitted == _]
                for _ in list(set(fitted)) if _ != -1]  # cell-ids of cells in one cluster
    ret = [[_ for _ in (get_from_cell(cell, head, body) for cell in cells)] for cells in clusters]
    # for cells in clusters:  # cluster consists of cells, for cluster in clusters
    #    tmp = []
    #    for cell in cells:  # for cell in a cluster of cells
    #        tmp.extend(get_from_cell(cell, head, body))  # find particles ids in the cell and add to tmp
    #    ret.append(tmp)
    return ret  # return the particle ids


# For usage of cuda.jit, run `conda install cudatoolkit=9.0` after the
# installation of Anaconda env. numba.cuda cannot compile under
# latest `cudatoolkit=9.2' currently.


@cuda.jit('float64(float64[:], float64[:], float64[:])', device=True)
def pbc_dist_cu(a, b, box):
    tmp = 0
    for i in range(a.shape[0]):
        d = b[i] - a[i]
        d = d - floor(d / box[i] + 0.5) * box[i]
        tmp += d * d
    return sqrt(tmp)


@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64[:,:])')
def pbc_pairwise_distance(x, y, box, ret):
    i, j = cuda.grid(2)
    if i >= x.shape[0] or j >= y.shape[0]:
        return
    if j >= i:
        return
    r = pbc_dist_cu(x[i], y[j], box)
    ret[j, i] = r
    ret[i, j] = r


@cuda.jit('void(float64[:,:], float64[:], float64[:])')
def pbc_pdist(x, box, ret):
    r"""pdist gpu ver with pbc distance metric.

    In [1]: k = 0

    In [2]: m = 5

    In [3]: for i in range(m-1):
        ...:     for j in range(i+1, m):
        ...:         k += 1
        ...:         print(k, i, j, i*m+j-(i+1)*i/2-i)
    1 0 1 1.0
    2 0 2 2.0
    3 0 3 3.0
    4 0 4 4.0
    5 1 2 5.0
    6 1 3 6.0
    7 1 4 7.0
    8 2 3 8.0
    9 2 4 9.0
    10 3 4 10.0

    :param x: np.ndarray, (n_coordinates, n_dimensions)
    :param box: np.ndarray, (n_dimensions,)
    :param ret: np.ndarray, see `scipy.spatial.distance.pdist` with nC2 elements.
    :return:
    """
    i = cuda.grid(1)
    if i >= x.shape[0] - 1:
        return
    for j in range(i + 1, x.shape[0]):
        r = pbc_dist_cu(x[i], x[j], box)
        ret[int(i * x.shape[0] + j - (i + 1) * i / 2 - i - 1)] = r
        # u_tri matrix, remove (i+1)i/2+i elements for the ith row.
