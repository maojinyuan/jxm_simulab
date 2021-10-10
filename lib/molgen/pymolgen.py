import numpy as np

class LinkCellList(object):
    def __init__(self, sc, box):
        self.ibox = np.floor(box / sc).astype(int)
        self.box  = box
        self.mul = np.array([1, self.ibox[1], self.ibox[1]*self.ibox[2]])
        self.vbox = np.array([np.arange(_) for _ in self.ibox])
        self.grid = np.roll(np.asarray(np.meshgrid(self.vbox[0], self.vbox[1], self.vbox[2])).reshape(3, -1).swapaxes(0, 1), 1, axis=-1)
        self.grid_icell = (self.grid * self.mul).sum(axis=-1)
        self.grid_neigh_index = np.asarray(np.meshgrid([0, 1, -1], [0, 1, -1], [0, 1, -1])).reshape(3, -1).swapaxes(0, 1)[1:]
        self.grid_neigh = np.array([self.ixcell(self.grid_neigh_index + _) for _ in self.grid], dtype=int)

        self.particle_icell = []

    def setParticle(self, pos):
        self.particle_icell = self.pcell(pos).tolist()

    def addParticle(self, p):
        p = np.asarray(p)
        if p.ndim == 1:
            self.particle_icell.append(self.pcell(p))
        elif p.ndim == 2:
                [self.particle_icell.append(self.pcell(_))  for _ in p]

    def getParticleNeigh(self, p):
        icell = self.pcell(p)
        neigh_index = self.grid_neigh[icell]
        neigh = []
        for n in neigh_index:
            neigh.extend(np.argwhere(self.particle_icell == n).flatten())
        return np.asarray(neigh, dtype=int)
            
    def picell(self, p):
        '''
        note : calculate the icell of one particle or particles with ndim = 1 or 2
        param: p, the position of particle(s), ndim = 1 or 2
        ret->: int or int ndarray 
        '''
        return (np.floor((p / self.box + 0.5) * self.ibox).astype(int) * self.mul).sum(axis=-1)

    def gicell(self, g):
        '''
        param: g : the grid icell with (ix, iy, iz), ndim = 1 or 2
        return-> 
        '''
        return (((g + self.ibox) % self.ibox) * self.mul).sum(axis=-1)

    def gvcell(self, g):
        '''
        param: g: the grid icell with (ix, iy, iz), ndim = 1 or 2
        return-> int, grid cell
        '''
        return (((g + self.ibox) % self.ibox) * self.mul).sum(axis=-1).astype(int)

def pbc(a, box):
    return a - box * np.round(a/box) 

def check_neardis(p, pset, box, dcric):
    if len(pset) == 0:
        return True
    else:
        v = pbc(pset - p, box)
        d = np.linalg.norm(v, axis=-1)
        if (d > dcric).all():
            return True
    return False

class xml_parser(object):
    def __init__(self, filename, needed=[]):
        from io import StringIO
        from xml.etree import cElementTree
        from pandas import read_csv

        tree = cElementTree.ElementTree(file=filename)
        root = tree.getroot()
        c = root[0]
        self.nodes = {}
        self.config = c.attrib
        for e in c:
            if e.tag == 'box':
                self.box = np.array([float(e.attrib['lx']), float(e.attrib['ly']), float(e.attrib['lz'])])
                continue
            if ((len(needed) != 0) and (e.tag not in needed)):
                continue
            if e.attrib['num'] == '0':
                continue
            self.nodes[e.tag] = read_csv(StringIO(e.text), delim_whitespace=True, squeeze=1, header=None).values

        # add
        if 'bond' in self.nodes.keys():
            self.nodes['bond'] = np.c_[self.nodes['bond'][:, 0], np.ones(self.nodes['bond'].shape[0]), self.nodes['bond'][:, 1:]]

class Molecule(object):
    def __init__(self, natoms, fname=None):
        self.natoms = int(natoms) 
        self.nbonds = 0
        self.nangles = 0
        self.ndihedrals = 0
        self.__fname = fname

        self.nodes = {}
        self.nodes['position'] = np.array([], dtype=float).reshape(0, 3)
        self.nodes['image'] = np.array([], dtype=int).reshape(0, 3)
        self.nodes['type'] = np.array([], dtype=str)
        self.nodes['mass'] = np.ones(self.natoms, dtype=float) 
        self.nodes['body'] = np.zeros(self.natoms, dtype=int) -1
        self.nodes['bond'] = np.array([]).reshape(0, 4)
        self.nodes['angle'] = np.array([]).reshape(0, 5)
        self.nodes['dihedral'] = np.array([]).reshape(0, 6)

        self.read(self.__fname, delelte_name=None, delelte_index=None)

    def __str__(self):
        return  "Molecule: [natoms-%d] [nbonds-%d]"%(self.natoms, self.nbonds)

    def __repr__(self):
        return  "Molecule: [natoms-%d] [nbonds-%d]"%(self.natoms, self.nbonds)

    def read(self, fn, delelte_name=None, delelte_index=None):
        if fn == None:
            return
        obj = xml_parser(fn)
        assert self.natoms >= int(obj.config['natoms']), '***Error!'
        assert delelte_name == None and delelte_index == None, "***Error! You cannot assing with name and index"

        # delete
        if delelte_name != None:
            delname = delelte_name
            delindexbool = (obj.nodes['type'] == delname)
            delindex     = np.arange(obj.config['natoms'])[delindexbool]
            for key, value in obj.nodes.items():
                if key in ['bond', 'angle', 'dihedral']:
                    self.nodes[key] = value[delindexbool]
                else:
                    self.nodes[key] = value[delindexbool]
                # need rewrite!

        elif delelte_index != None:
            delindex = delelte_index
        else:
            for key, value in obj.nodes.items():
                self.nodes[key] = value
            

    def setParticleTypes(self, types):
        assert type(types) == str, 'Molecule.setParticleTypes Error!'
        _types = np.array([_.strip() for _ in types.split(',')], dtype=str)
        assert self.nodes['type'].shape[0] + len(_types)  == self.natoms, "***InitError! The number of particle types %d is different from the initialized particle number %d"%(self.nodes['type'].shape[0], self.natoms)
        self.nodes['type'] = np.r_[self.nodes['type'], _types]

    def setTopology(self, bondstr):
        _bond = np.array([[int(u.split('-')[0]), int(u.split('-')[1])] for u in bondstr.split(',')], dtype=int)
        bond = np.array([[self.nodes['type'][u[0]] + '-' + self.nodes['type'][u[1]], 1.0, u[0], u[1]] for u in _bond], dtype=object)
        self.nodes['bond'] = np.r_[self.nodes['bond'], bond]
        self.nbonds += self.nodes['bond'].shape[0]

    def setBondLength(self, length, a=None, b=None):
        assert type(length) in [str, int, float], "***Error! The bond length should be float nubmer"
        value = float(length)

        if a == None:
            self.nodes['bond'][:, 1] = value
        else:
            bondtype = a + '-' + b 
            self.nodes['bond'][:, 1][self.nodes['bond'][:, 0] == bondtype] = value 

    def setAngleDegree(self, degree, a=None, b=None, c=None):
        assert type(degree) in [str, int, float], "***Error! The degree should be float nubmer"
        value = float(degree)
        if a == None:
            self.nodes['angle'] = value 
        else:
            tp = '-'.join[a, b, c]
            self.nodes['angle'][:, 1][self.nodes['angle'][:, 0] == tp] = value 

    def setDihedralDegree(self, degree, a=None, b=None, c=None, d=None):
        assert type(degree) in [str, int, float], "***Error! The degree should be float nubmer"
        value = float(degree)
        if a == None:
            self.nodes['angle'] = value 
        else:
            tp = '-'.join[a, b, c, d]
            self.nodes['angle'][:, 1][self.nodes['angle'][:, 0] == tp] = value 
        
    def setMass(self, mass, name=None, index=None):
        assert type(mass) in [str, int, float], "***Error! The mass should be float nubmer"
        assert name == None and index == None, "***Error! You cannot assing with name and index"

        value = float(mass)
        if name != None:
            self.nodes['mass'][self.nodes['type'] == name] = value
        elif index != None:
            self.nodes['mass'][index] = value
        else:
            self.nodes['mass'] = value
        
    def setBox(self, lx, ly, lz, mlx=None, mly=None, mlz=None):
        if mlx == None:
            xli, yli, zli, xhi, yhi, zhi = -lx/2.0, -ly/2.0, -lz/2.0, lx/2.0, ly/2.0, lz/2.0 
        else:
            xli, yli, zli, xhi, yhi, zhi = lx, ly, lz, mlx, mly, mlz

        self.edge = np.array([xli, yli, zli, xhi, yhi, zhi], dtype=float)
        self.box = self.edge[3:] - self.edge[:3]

    def setSphere(sx, sy, sz, r_main, r_max):
        pass

    def setCylinder():
        pass

    def setBody(self, value, name, index):
        node = 'body'
        assert type(value) in [str, int, float], "***Error! The mass should be float nubmer"
        assert name == None and index == None, "***Error! You cannot assignng with name and index"

        value = float(mass)
        if name != None:
            self.nodes[node][self.nodes['type'] == name] = value
        elif index != None:
            self.nodes[node][index] = value
        else:
            self.nodes[node] = value

    def setBodyEvacuaton():
        self.BodyEvacuaton = True
 
    def setMinimumDistance(self, min_dis, na=None, nb=None):
        pass

    def generate(self, g):
        bound = 0
        box = self.box if 'box' in self.__dict__ else g.box

        # generate pos
        while True:
            for i in range(self.natoms):
                x = (np.random.random() - 0.5) * box[0]
                y = (np.random.random() - 0.5) * box[1]
                z = (np.random.random() - 0.5) * box[2]
                p = np.array([x, y, z])
            break

class Generators():
    def __init__(self, lx, ly, lz):
        self.box = np.array([lx, ly, lz])
        self.natoms = 0
        self.nbonds = 0
        self.nangles = 0
        self.ndihedrals = 0
        self.molecules = []
        self.nmolecules  = [] 
        self.nmols = 0

        self.status_generated = np.zeros(self.natoms, dtype=bool)

    def addMolecule(self, mol, nmol):
        self.nbonds += mol.nbonds * nmol
        self.natoms += mol.natoms * nmol
        self.nangles += mol.nangles * nmol
        self.ndihedrals += mol.ndihedrals * nmol
        self.nmols += nmol
        self.nmolecules.append(nmol)
        self.molecules.append(mol)

    def outPutXml(self, outname, needed=[], config=None):
        if config == None:
            config = {'time_step': 0, 'dimensions':3, 'natoms': self.natoms}

        if len(needed) == 0:
            needed = self.nodes.keys()

        o = open(outname+'.xml', 'w')
        formatter = {
            'position': '{:<18.8f}{:<18.8f}{:<18.8f}\n',
            'image': '{:<d} {:<d} {:<d}\n',
            'bond': '{:<s} {:<d} {:<d}\n',
            'angle': '{:<s} {:<d} {:<d} {:<d}\n',
            'dihedral': '{:<s} {:<d} {:<d} {:<d} {:<d}\n',
            'type': '{:<s}\n',
            'body': '{:<d}\n',
            'h_init': '{:<d}\n',
            'h_cris': '{:<d}\n',
            'mass': '{:<.4f}\n',
        }

        # 'write headers'
        o.write('<?xml version="1.0" encoding="UTF-8"?>\n<galamost_xml version="1.6">\n')
        o.write('<configuration time_step="%s" dimensions="%s" natoms="%s">\n' % (config['time_step'], config['dimensions'], config['natoms']))
        o.write('<box lx="%f" ly="%f" lz="%f" xy="0" xz="0" yz="0"/>\n' % (self.box[0], self.box[1], self.box[2]))

        for node  in needed:
            value = self.nodes[node]

            o.write('<%s num="%d">\n' % (node, len(value)))

            if node in ['bond', 'angle', 'dihedral']:
                for p in value: o.write(formatter[node].format(p[0], *p[2:]))
            else:
                for p in value: o.write(formatter[node].format(*p))

            o.write('</%s>\n' % (node))

        o.write('</configuration>\n</galamost_xml>\n')
        o.close()

    def genMolecule(self):
        # initialize
        self.nodes = {}
        self.nodes['position'] = np.zeros((self.natoms, 3), dtype=float)
        self.nodes['image']   = np.zeros((self.natoms, 3), dtype=int)
        self.nodes['bond'] = np.array([['bond', 1.0, 0, 0]] * self.nbonds, dtype=object)
        self.nodes['angle'] = np.array([['angle', 1.0, 0, 0, 0]] * self.nangles, dtype=object)
        self.nodes['dihedral'] = np.array([['dihedral', 1.0, 0, 0, 0, 0]] * self.ndihedrals, dtype=object)
        self.nodes['type']   = np.array(['A'] * self.natoms, dtype=str)

        # for mol, nmol in zip(self.mols, self.nmols):
            # for i in range(nmol):
                # mol.generate(self)
    
# main
if __name__ == '__main__':
    mol0 = Molecule(20, 'init.xml')
    mol0.setParticleTypes('A, A, B, C, D')
    mol0.setTopology('15-16, 16-17, 17-18, 18-19')
    mol0.setBondLength(2.0, 'A', 'A')

    mol1 = Molecule(5)
    mol1.setParticleTypes('A, B, B, C, C')
    mol1.setBox(20, 20, 20)
    mol1.setMass(1.0)

    gen = Generators(40, 40, 40)
    gen.addMolecule(mol0, 10)
    gen.addMolecule(mol1, 10)
    gen.genMolecule()
    gen.outPutXml('o', needed=['position', 'type', 'image', 'bond', 'angle', 'dihedral'])
