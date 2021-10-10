import numpy as np


def bond_hash_dualdirect(bond, natoms):
    """
    :param bond: bond data in hoomdxml format (name, id1, id2)
    :param natoms: total number of particles
    :return: hash table of with value in {bondname1: [idxes], bondname2:[idxes]...} for each particle (in dual direct)
    """
    bond_hash_nn = {}
    print('Building bond hash...')
    if not isinstance(bond, np.ndarray):
        return {}
    for i in range(natoms):
        bond_hash_nn[i] = []
    for b in bond:
        idx = b[1]
        jdx = b[2]
        bond_hash_nn[idx].append(jdx)
        bond_hash_nn[jdx].append(idx)
    # print('Done.')
    return bond_hash_nn
