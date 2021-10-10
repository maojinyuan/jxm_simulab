import numpy as np

def pbc(a, box):
    return a - box * np.round(a/box)

def eigh(a, box, com):
    '''
    :para a: ndims=2, several points
    :para com: center of points
    '''
    a = a - com
    v = a - box * np.round(a/box)
    return np.linalg.eigh(v.T.dot(v)/v.shape[0])
