import time
from cmath import exp

import numba as nb
import numpy as np
import tqdm
from numba import float64, complex128


@nb.jit(nopython=True)
def _q_vec(n_q, q, dq, rtol=1e-3):
    r"""
    :param n_q: a list of int: (n_q_x, n_q_y, ...) shape of the mesh grid
    :param q: float, modulus of q
    :param dq: float, n_q[0] / 2 * dq ~ q
    :param rtol: float, relative tolerance
    :return: list, list of q vectors
    """
    ret = []
    for q_ary in np.ndindex(n_q):
        q_tmp = 0
        for qi in q_ary:
            q_tmp += (qi - n_q[0] / 2) ** 2
        q_tmp = q_tmp ** 0.5 * dq
        if abs(q_tmp - q) / q < rtol:
            ret.append(q_ary)
    return ret


@nb.guvectorize([(float64[:, :], float64[:, :], complex128[:])],
                '(n, p),(m, p)->(n)', target='parallel')
def exp_iqr(a, b, ret):
    for i in range(a.shape[0]):
        tmp = 0
        for j in range(b.shape[0]):
            prod = 0
            for k in range(b.shape[1]):
                prod += a[i, k] * b[j, k]
            tmp += exp(-1j * prod)
        ret[i] = tmp


def incoherent_scattering(traj, q, dq, rtol=1e-3, q_vectors=None):
    n_dim = traj.shape[-1]
    n_q = (int(np.round(q / dq)) * 2,) * n_dim  # this is not round, it's ceil
    s = time.time()
    q_vecs = q_vectors
    if q_vecs is None:
        q_vecs = (np.asarray(_q_vec(n_q, q, dq, rtol), dtype=np.float) - n_q[0] / 2) * dq
    n_frames = traj.shape[0]
    corr = np.zeros((n_frames, traj.shape[1]), dtype=np.complex128)
    if q_vecs.shape[0] > 1000:
        raise ValueError("Too many q vectors! %d > 1000" % q_vecs.shape[0])
    print("Generate q vecs in %.6fs, processing with num of q vectors: %d" % (time.time() - s, q_vecs.shape[0]))
    for q_vec in tqdm.tqdm(q_vecs, desc="Processing with Q vectors", unit=r"Q vectors", ncols=100):
        traj_tmp = exp_iqr(traj, [q_vec])
        corr = corr + np.fft.ifft(np.abs(np.fft.fft(traj_tmp, axis=0, n=2 * n_frames)) ** 2,
                                  axis=0, n=2 * n_frames)[:n_frames]
    # traj = exp_iqr(traj, q_vecs) / q_vecs.shape[0]
    # corr = np.fft.ifft(np.abs(np.fft.fft(traj, axis=0, n=2*n_frames))**2, axis=0, n=2*n_frames)[:n_frames]
    corr = corr / np.arange(n_frames, 0, -1)[:, None] / q_vecs.shape[0]
    return corr, q_vecs
