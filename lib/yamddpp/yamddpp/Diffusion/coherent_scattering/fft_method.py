import numpy as np


def coherent_scattering(traj, n_bins):
    min_r, max_r = traj.min(), traj.max()
    n_frames, n_dim = traj.shape[0], traj.shape[-1]
    frho_t = np.empty((n_frames, *(n_bins,) * n_dim))
    for i, frame in enumerate(traj):
        frho_t[i], _ = np.fft.fftn(np.histogramdd(frame, bins=(n_bins,) * n_dim, range=[[min_r, max_r]] * n_dim))
    corr = np.fft.ifft(np.abs(np.fft.fft(frho_t, axis=0, n=2 * n_frames)) ** 2, axis=0)[:n_frames]
    return corr / np.expand_dims(np.arange(n_frames, 0, -1), axis=tuple(np.arange(n_dim) + 1))
