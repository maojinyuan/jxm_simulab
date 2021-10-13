import os, sys, time
import numpy as np
import pandas as pd
import linecache
import multiprocessing as mp

class MPI_tackle(object):
    """ A class wrapper for multiprocessing calculations
    
    This class should be inherited, then add deliverWork function.
    And some preparatons for deliverwork can be wrapped into a preWork function.
    
    Note1: The multiprocess package can not be used for distributed nodes.         
    """
    def __init__(self, data, nps=None, chunksize_data=None):
        self.data = data
        self.count = 0
        self.nps = nps if nps != None else 8
        self.chunksize_data = chunksize_data

    def start(self):
        if self.chunksize_data == None:
            self.pool = mp.Pool(self.nps)
            self.pool.map(self.deliverWork, self.data)
            self.pool.close()
            self.pool.join()
        else:
            n, _ = divmod(len(self.data), self.chunksize_data)
            for i in range(n):
                s = self.chunksize_data * i
                e = self.chunksize_data * (i + 1) 
                idata = self.data[s:e]

                # start pool and calculation part-i
                self.pool = mp.Pool(self.nps)
                self.pool.map(self.deliverWork, idata)
                self.pool.close()
                self.pool.join()
                time.sleep(3)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

class GetSimilarity(MPI_tackle):
    """ Class inherited from MPI_tackle for the calculations of mol similarity.
    
    add [deliverWork], [preWork], and helper function [getSimilarity]
    """
    def deliverWork(self, fn):
        targets = pd.read_csv(fn, header=None, sep=' ').values
        rr      = [_[0] + ',' + self.getSimilarity(_[1]) for _ in targets]

        if self.outpath == 'origin':
            outname = os.path.normpath(os.path.splitext(fn)[0] + '.fsi')
        else:
            outname = os.path.normpath(self.outpath + os.path.basename(fn).split('.')[0] + '.fsi')

        np.savetxt(outname, rr, fmt='%s')

    def preWork(self, **kwargs):
        assert 'B2AR' in kwargs.keys(), print("Error! wrong parameters") 
        suppl = Chem.SmilesMolSupplier(kwargs['B2AR'], ' ', 0, 1, False, True)
        mols = [x for x in suppl if x is not None]
        self.fps = [Chem.RDKFingerprint(x) for x in mols]

    def getSimilarity(self, _smi):
        try:
            mol = Chem.MolFromSmiles(_smi)
            fp  = Chem.RDKFingerprint(mol)
            similarity = [DataStructs.FingerprintSimilarity(fp, _) for _ in self.fps]
            maxv = max(similarity)
            maxv_arg = similarity.index(maxv) + 1
            meanv =  sum(similarity) / len(similarity)
            sim = str(maxv) + "," + str(maxv_arg) + "," + str(meanv) + "," + _smi
            return sim
        except:
            return 'nan' 

class GetSmilesFromPdbqt(MPI_tackle):
    """Class inherited from MPI_tackle for get smiles info from pdbqt file.
    
    Be careful, too many mp processes and I/O operation may cause out/input Error. 
    """
    def deliverWork(self, path):
        fnames = os.popen('ls %s/*/*.pdbqt'%(path)).read().strip().split('\n')
        rr = []
        for _ in  fnames:
            try:
                rr.append(_ + " " + linecache.getline(_, 3).split(':')[1].strip())
            except:
                pass

        if self.outpath == 'origin':
            outname = os.path.normpath(path + os.path.basename(path) + '.smi')
        else:
            outname = os.path.normpath(self.outpath + os.path.basename(path) + '.smi')

        np.savetxt(outname, rr, fmt='%s')

    def preWork(self, **kwargs):
        assert 'outpath' in kwargs.keys(), print("Error! wrong parameters") 
        self.outpath = kwargs['outpath']

def ComputeSimilarity(): 
    data_path = '../input-files/'
    names = [data_path + _ for _ in os.listdir(data_path)] 
    outpath = 'origin'
    kwargs = {'B2AR':'B2AR-small-molecule.smi', 'outpath':outpath}

    S = GetSimilarity(names, nps=int(sys.argv[1]))
    S.preWork(**kwargs)
    S.start()

def ComputeGetInfo():
    data_path = '../input-files/'
    names = [data_path + _ for _ in os.listdir(data_path)] 
    outpath = 'origin'
    kwargs = {'outpath': outpath}

    S = GetSmilesFromPdbqt(names, nps=int(sys.argv[1]))
    S.preWork(**kwargs)
    S.start()

if __name__ == '__main__':
    # ComputeSimilarity()
    ComputeGetInfo()