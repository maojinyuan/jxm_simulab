from mpi4py import MPI
import os, sys
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

def getSimilarity(_smi):
    try:
        mol = Chem.MolFromSmiles(_smi)
        fp  = Chem.RDKFingerprint(mol)
        similarity = [DataStructs.FingerprintSimilarity(fp, _) for _ in local_fps]
        maxv = max(similarity)
        maxv_arg = similarity.index(maxv) + 1
        meanv =  sum(similarity) / len(similarity)
        sim = str(maxv) + "," + str(maxv_arg) + "," + str(meanv) + "," + _smi
        return sim
    except:
        return 'nan' 

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

if comm_rank == 0:
    data_path = '../output/'
    data = [os.path.realpath(data_path + _) for _ in os.listdir(data_path)]
    suppl = Chem.SmilesMolSupplier('B2AR-small-molecule.smi', ' ', 0, 1, False, True)
    mols = [x for x in suppl if x is not None]
    fps = [Chem.RDKFingerprint(x) for x in mols]

local_data = comm.bcast(data if comm_rank == 0 else None, root=0)
local_fps  = comm.bcast(fps if comm_rank == 0 else None, root=0)

for i, fname in enumerate(local_data):
    irank, mod = divmod(i, comm_size)

    if irank == comm_rank:
        targets = pd.read_csv(fname, header=None, sep=' ').values
        rr      = [_[0] + ',' + getSimilarity(_[1]) for _ in targets]
        outname = os.path.splitext(fname)[0] + '.fsi'
        np.savetxt(outname, rr, fmt='%s')
