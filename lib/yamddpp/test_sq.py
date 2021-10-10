import numpy as np

from yamddpp.Scattering import rdf_xy

# x = np.random.random((100,3))
# y = np.random.random((100,3))

# np.savetxt('x.txt', x)
# np.savetxt('y.txt', y)

x = np.loadtxt('x.txt')
y = np.loadtxt('y.txt')

# ret = scatter_xy(x, y, x_range=[(0,1.)]*3, r_cut=0.3, q_bin=1, q_max=20, zero_padding=1, expand=0, use_gpu=3)

# ret2 = scatter_xy(x, y, x_range=[(0,1.)]*3, r_cut=0.3, q_bin=1, q_max=20, zero_padding=1, expand=0, use_gpu=False)
# print(ret, ret2)

ret1 = rdf_xy(x, y, x_range=[(0, 1)] * 3, bins=(100, 100, 100), use_gpu=False)
ret2 = rdf_xy(x, y, x_range=[(0, 1)] * 3, bins=(100, 100, 100), use_gpu=3)

print(ret1, ret2)
print(np.allclose(ret1, ret2))

x = np.loadtxt('x.txt') - 0.5
y = np.loadtxt('y.txt') - 0.5

# ret = scatter_xy(x, y, x_range=[(0,1.)]*3, r_cut=0.3, q_bin=1, q_max=20, zero_padding=1, expand=0, use_gpu=3)

# ret2 = scatter_xy(x, y, x_range=[(0,1.)]*3, r_cut=0.3, q_bin=1, q_max=20, zero_padding=1, expand=0, use_gpu=False)
# print(ret, ret2)

ret1 = rdf_xy(x, y, x_range=[(-.5, .5)] * 3, bins=(100, 100, 100), use_gpu=False)
ret2 = rdf_xy(x, y, x_range=[(-.5, .5)] * 3, bins=(100, 100, 100), use_gpu=3)

print(ret1, ret2)
print(np.allclose(ret1, ret2))
