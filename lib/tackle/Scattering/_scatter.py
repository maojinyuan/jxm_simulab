import numba as nb
import numpy as np 

@nb.jit(nopython=True)
def generate_qvec(nq, mq, dq, rtol=1e-3):
    ret = []
    for _q in np.ndindex(nq):
        q = np.array(_q, dtype=np.float32) * dq
        qnorm = np.linalg.norm(q) 
        if abs((qnorm - mq) / mq) < rtol:
            ret.append((q[0],  q[1], q[2]))
            ret.append((-q[0], q[1], q[2]))
            ret.append((q[0], -q[1], q[2]))
            ret.append((q[0],  q[1],-q[2]))

    return np.array(ret, dtype=np.float32) 


def IncoherentISF(u, nq, dq, mq=7.0, rtol=1e-3, qvec=None, cum=False, gpu=None):

    if qvec is None:
        qvec = generate_qvec(nq, mq, dq, rtol)

    if gpu is not None:
        import cupy as cp
        cp.cuda.Device(int(gpu)).use()

    # echo message
    print(">>> ISF-Parameters:")
    print("   qmain {:.2f} qvectors {:d}".format(mq, qvec.shape[0]))

    rr = []

    # calcuate isf via cupy.matmul 
    qvec = qvec.T if gpu is None else cp.asarray(qvec.T)

    for i in range(u.shape[0]):
        print("Frames: %d"%(i)) if i%100 == 0 else None

        if gpu is None:
            d = u[i] - u[0]
            theta = d.dot(qvec) 
        else:
            d = cp.asarray((u[i] - u[0]))
            theta = cp.asnumpy(cp.matmul(d_cu, qvec_cu))
            
        r = np.cos(theta)
        if cum == True:
            rr.append(r.mean())
        else:
            rr.append(r.mean(axis=-1))


    rr = np.asarray(rr)

    return rr 


def StaticStructureFactor(u, box, q_sparse=0.5, qmax=None, qbin=None):
    """Calculating static structure factor
    :param  u       : np.ndarray, positions
    :param  box     : np.ndarray
    :param  q_sparse: float, skip some grid points to accelarate calculations 
    :param  qmax    : float
    :param  qbin    : float, grid spacing in reciprocal space
    """

    if qmax is None: qmax = 160 * np.pi / box.min()
    if qbin is None: qbin = (2 * np.pi / box.min())

    # echo message
    print("Parameters: qmax=%.6f  qbin=%.6f  q_sparse=%.2f" % (qmax, qbin, q_sparse))

    kmax_x = kmax_y = kmax_z = int(qmax / (qbin))
    qvec = []
    qvec_binidx = []
    kmax_step = int(1 / q_sparse)
    for i in range(-kmax_x, kmax_x, kmax_step):
        for j in range(-kmax_y, kmax_y, kmax_step):
            for k in range(0, kmax_z, kmax_step):
                _q = 2 * np.pi * np.array([i, j, k]) / box
                _qabs = np.linalg.norm(_q)
                if qbin <= _qabs < qmax:
                    qvec.append(_q)
                    qvec_binidx.append(int(_qabs / qbin))

    qvec = np.asarray(qvec)
    qvec_cu = cp.asarray(qvec.T).astype(np.float32)
    qvec_binidx = np.asarray(qvec_binidx)

    q = np.arange(0, qmax, qbin)
    sq = np.zeros_like(q) 
    sq_count = np.zeros_like(q) 

    o = open('sq_cu.dat', 'w')
    
    if u.ndim == 3:
        for i in range(u.shape[0]):
            print("Frame: %d" % (i))
            p_cu = cp.asarray(u[i]).astype(np.float32)
            print(p_cu.shape, qvec_cu.shape)
            theta = cp.matmul(p_cu, qvec_cu)
            expikr = ((np.cos(theta).sum(axis=0))**2 + (np.sin(theta).sum(axis=0))**2) / p_cu.shape[0]
            for j in range(len(expikr)):
                sq[qvec_binidx[j]] += expikr[j]
                sq_count[qvec_binidx[j]] += 1
    elif u.ndim == 2:
        p_cu = cp.asarray(u[i]).astype(np.float32)
        theta = cp.matmul(p_cu, qvec_cu)
        expikr = ((np.cos(theta).sum(axis=0))**2 + (np.sin(theta).sum(axis=0))**2) / p_cu.shape[0]
        for j in range(len(expikr)):
            sq[qvec_binidx[j]] += expikr[j]
            sq_count[qvec_binidx[j]] += 1

    sq = np.asarray([sq[i] / sq_count[i] if sq_count[i] != 0 else 0.0 for i in range(sq_count.shape[0])]) 

    return np.c_[q, sq] 

if __name__ == '__main__':
    pass
