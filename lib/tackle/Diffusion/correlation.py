import numpy as np

def autocorrFFT(X):
	M=X.shape[0]
	Fx = np.fft.rfft(X.T[0], 2*M)
	Fy = np.fft.rfft(X.T[1], 2*M)
	Fz = np.fft.rfft(X.T[2], 2*M)
	corr = abs(Fx)**2 + abs(Fy)**2 + abs(Fz)**2
	res = np.fft.irfft(corr)
	res= (res[:M]).real
	return res

def msd_fft(X):
	M = X.shape[0]
	D = np.square(X).sum(axis=1)
	D = np.append(D, 0)
	S2 = autocorrFFT(X)
	S2 = S2 / np.arange(M, 0, -1)
	Q = 2 * D.sum()
	S1 = np.zeros(M)
	for m in range(M):
		Q = Q - D[m-1] -D[M-m]
		S1[m] = Q / (M-m)
	return S1 - 2 * S2


