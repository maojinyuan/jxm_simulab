from math import floor, atan2, sqrt, pi

import numpy as np
from numba import cuda, void, float64, float32, complex128, complex64, int32

from ._spherical_harmonics import gen_sph
from ..plists import nlist


class ql:
    def __init__(self, frame, ls=np.asarray([4, 6]), cell_guess=15, n_guess=10):
        self.frame = frame
        self.gpu = frame.gpu
        self.ls = ls
        self._n_m = int(2 * np.max(ls) + 1)  # -l ~ 0 ~ l
        self._n_l = ls.shape[0]  # number of l
        self.nlist = nlist(self.frame, contain_self=1,
                           cell_guess=cell_guess, n_guess=n_guess)
        self.r_cut = frame.r_cut
        self.dtype = self.frame.x.dtype
        self.ql_local = None
        self.ql_avg = None
        global sphHar
        sphHar = gen_sph(self.frame.x.dtype)
        if self.dtype == np.dtype(np.float64):
            self.float = float64
            self.complex = complex128
        else:
            self.float = float32
            self.complex = complex64
        self.cu_ql_local = self._ql_local_func()
        self.cu_ql_avg = self._ql_avg_func()
        self.n_bonds = None

    def update(self, x=None, box=None, rc=None, mode='all'):
        if x is not None:
            self.frame.x = x
        if box is not None:
            self.frame.box = box
        if rc is not None:
            self.frame.r_cut = rc
            self.r_cut = rc
        self.frame.update()
        self.nlist.update()
        self.calculate(mode)

    def calculate(self, mode='all'):
        with cuda.gpus[self.gpu]:
            d_ls = cuda.to_device(self.ls)
            device = cuda.get_current_device()
            tpb = device.WARP_SIZE
            bpg = int(np.ceil(self.frame.x.shape[0] / tpb))
            if mode == 'all' or mode == 'local':
                self.ql_local = np.zeros((self.frame.x.shape[0], self._n_l),
                                         dtype=self.frame.x.dtype)
                d_ql_local = cuda.to_device(self.ql_local)
                self.cu_ql_local[bpg, tpb](
                    self.frame.d_x,
                    self.frame.d_box,
                    self.frame.r_cut,
                    self.nlist.d_nl,
                    self.nlist.d_nc,
                    d_ls,
                    d_ql_local
                )
                d_ql_local.copy_to_host(self.ql_local)
                cuda.synchronize()
            if mode == 'all' or mode == 'avg':
                self.ql_avg = np.zeros(self.ls.shape[0])
                q_vec_real = np.zeros((self.ls.shape[0], self._n_m),
                                      dtype=self.frame.x.dtype)
                q_vec_imag = np.zeros((self.ls.shape[0], self._n_m),
                                      dtype=self.frame.x.dtype)
                d_qvec_real = cuda.to_device(q_vec_real)
                d_qvec_imag = cuda.to_device(q_vec_imag)
                n_bonds = np.zeros(1, dtype=np.int32)
                d_n_bonds = cuda.to_device(n_bonds)
                self.cu_ql_avg[bpg, tpb](
                    self.frame.d_x,
                    self.frame.d_box,
                    self.frame.r_cut,
                    self.nlist.d_nl,
                    self.nlist.d_nc,
                    self.ls,
                    d_qvec_real,
                    d_qvec_imag,
                    d_n_bonds
                )
                d_n_bonds.copy_to_host(n_bonds)
                d_qvec_real.copy_to_host(q_vec_real)
                d_qvec_imag.copy_to_host(q_vec_imag)
                cuda.synchronize()
                q_vec = q_vec_real + 1j * q_vec_imag
                self.n_bonds = n_bonds[0]
                if self.n_bonds < 1.0:
                    self.n_bonds = 1.0
                for i in range(q_vec.shape[0]):
                    tmp = 0
                    for j in range(q_vec.shape[1]):
                        tmp += abs(q_vec[i, j] / self.n_bonds) ** 2
                    self.ql_avg[i] = sqrt(tmp * 4 * np.pi / (2 * self.ls[i] + 1))

    def _ql_local_func(self):
        _qvi = (self._n_l, self._n_m)
        _rei = (self._n_l,)
        nb_complex = self.complex
        nb_float = self.float

        @cuda.jit(void(self.float[:, :], self.float[:], self.float,
                       int32[:, :], int32[:], int32[:], self.float[:, :]))
        def _ql_local(x, box, rc, nl, nc, ls, ret):
            i = cuda.grid(1)
            if i >= x.shape[0]:
                return
            q_vec_i = cuda.local.array(_qvi, nb_complex)
            res_i = cuda.local.array(_rei, nb_float)
            for _ in range(q_vec_i.shape[0]):
                res_i[_] = 0
                for __ in range(q_vec_i.shape[1]):
                    q_vec_i[_, __] = 0 + 0j
            nn = 0.0
            for j in range(nc[i] - 1):
                pj = nl[i, j]
                for k in range(j + 1, nc[i]):
                    pk = nl[i, k]
                    dx = x[pk, 0] - x[pj, 0]
                    dy = x[pk, 1] - x[pj, 1]
                    dz = x[pk, 2] - x[pj, 2]
                    dx = dx - box[0] * floor(dx / box[0] + 0.5)
                    dy = dy - box[1] * floor(dy / box[1] + 0.5)
                    dz = dz - box[2] * floor(dz / box[2] + 0.5)
                    dr = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    if dr >= rc:
                        continue
                    nn += 1.0
                    phi = atan2(dy, dx)
                    if phi < 0:
                        phi = phi + 2 * pi
                    cosTheta = dz / dr
                    for _l in range(ls.shape[0]):
                        l = ls[_l]
                        for m in range(-l, l + 1):
                            q_vec_i[_l, m + l] += sphHar(l, m, cosTheta, phi)
            # print(i, nn)
            if nn < 1.0:
                nn = 1.0
            for _ in range(q_vec_i.shape[0]):
                for __ in range(q_vec_i.shape[1]):
                    res_i[_] += abs(q_vec_i[_, __] / nn) ** 2
            for _ in range(q_vec_i.shape[0]):
                ret[i, _] = sqrt(res_i[_] * 4 * pi / (2 * ls[_] + 1))

        return _ql_local

    def _ql_avg_func(self):

        @cuda.jit(
            void(self.float[:, :], self.float[:], self.float, int32[:, :], int32[:],
                 int32[:], self.float[:, :], self.float[:, :], int32[:]))
        def _ql_avg(x, box, rc, nl, nc, ls, q_vec_real, q_vec_imag, n_bonds):
            i = cuda.grid(1)
            if i >= x.shape[0]:
                return
            nn = 0
            for j in range(nc[i]):
                pj = nl[i, j]
                if pj <= i:
                    continue
                dx = x[pj, 0] - x[i, 0]
                dy = x[pj, 1] - x[i, 1]
                dz = x[pj, 2] - x[i, 2]
                dx = dx - box[0] * floor(dx / box[0] + 0.5)
                dy = dy - box[1] * floor(dy / box[1] + 0.5)
                dz = dz - box[2] * floor(dz / box[2] + 0.5)
                dr = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                if dr >= rc:
                    continue
                nn += 1
                phi = atan2(dy, dx)
                if phi < 0:
                    phi = phi + 2 * pi
                cosTheta = dz / dr
                for _l in range(ls.shape[0]):
                    l = ls[_l]
                    for m in range(-l, l + 1):
                        tmp = sphHar(l, m, cosTheta, phi)
                        cuda.atomic.add(q_vec_real[_l], m + l, tmp.real)
                        cuda.atomic.add(q_vec_imag[_l], m + l, tmp.imag)
                        # use very small arrays.
                        # qvec[i, _l, m + l] += sphHar(l, m, cosTheta, phi)  # thread-safe
            cuda.atomic.add(n_bonds, 0, nn)

        return _ql_avg
