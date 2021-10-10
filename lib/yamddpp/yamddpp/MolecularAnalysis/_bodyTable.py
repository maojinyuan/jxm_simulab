import numpy as np


def body_hash(body):
    """
    :param body: 1-D np array with n atoms, for bodies (-1 for not a body)
    :return:
    """
    ret = {}
    print('Build body hash...')
    natoms = len(body)
    bodies = list(set(list(body)))
    bodies.remove(-1)
    idxes = np.arange(natoms)
    for b in bodies:
        ret[b] = idxes[body == b]
    print('Done.')
    return ret
