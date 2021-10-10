#!/usr/bin/env python
import sys
import numpy as np
from xml.etree import cElementTree  # pypy will be a bit slower than python
from pandas import read_csv
from io import StringIO
from itertools import chain

class System():
    def __init__(self):
        self.__str__ = '~'
        self.ncharges = 0.0

class residue_type(object):
    def __init__(self, _name, _atoms, _atom_charges, _status_atom_charges, _atom_types, _bonds, _bond_parameters, _status_bond_parameters, _angles, _angle_parameters, _status_angles,_dihs, _dih_parameters, _status_dihs, _head, _tail):
        self.name = _name 
        self.atoms = _atoms
        self.atom_types = _atom_types

        # for charges
        self.atom_charges = _atom_charges
        self.status_atom_charges = _status_atom_charges
        
        # for bonds
        self.bonds = np.array(_bonds).astype(np.int)
        self.status_bond_parameters = _status_bond_parameters
        self.bond_parameters = _bond_parameters

        # for angles
        self.angles = _angles
        self.status_angles = _status_angles
        self.angle_parameters = _angle_parameters

        # for dihs
        self.dihs = _dihs
        self.status_dihs = _status_dihs
        self.dih_parameters = _dih_parameters

        self.head = _head
        self.tail = _tail

        self.natoms = len(self.atoms)

    def get_linkbond(self, _index):
        return self.bonds[np.argwhere(self.bonds == _index)[:, 0]]

class force_field:
    def __init__(self, _name):
        self.name = _name
    class cls_atomtypes:
        pass
    class cls_bondtypes:
        pass
    class cls_angletypes:
        pass
    class cls_dihedraltypes:
        pass

class hoomd_xml(object):
    def get_res(self, name, head, tail, terhead, tertail, head_atom_type, tail_atom_type, head_charge, tail_charge):
        res = self.nodes[name]
        atoms = res['atoms'][:, 1]
        natoms = len(atoms)
        atom_types = res['atoms'][:, 2]
        if res['atoms'].shape[1] == 4:
            atom_charges = res['atoms'][:,3]
            status_atom_charges = True
        else:
            atom_charges = None 
            status_atom_charges = False

        delete_index = []
        if head != None:
            index_head = atoms.tolist().index(head)
            atom_charges[index_head] = head_charge
            delete_index.append(atoms.tolist().index(terhead))
            atom_types[atoms.tolist().index(head)] = head_atom_type
        if tail != None:
            index_tail = atoms.tolist().index(tail)
            atom_charges[index_tail] = tail_charge
            delete_index.append(atoms.tolist().index(tertail))
            atom_types[atoms.tolist().index(tail)] = tail_atom_type
        new_atoms = [] 
        new_atom_charges = []
        new_atom_types = [] 
        for i in range(natoms):
            if i not in delete_index:
                new_atoms.append(atoms[i])
                new_atom_types.append(atom_types[i])
                if status_atom_charges == True:
                    new_atom_charges.append(atom_charges[i])

        # for bonds
        bonds = res['bonds']
        bond_parameters = res['bond_parameters']
        new_bonds = []
        new_bond_parameters = []
        for i, bd in enumerate(bonds):
            a, b = bd
            atom_a, atom_b = atoms[a], atoms[b]
            atom_type_a, atom_type_b = atom_types[a], atom_types[b]
            if (a in delete_index) or (b in delete_index):
                pass
            else:
                new_a = new_atoms.index(atom_a) 
                new_b = new_atoms.index(atom_b) 
                new_bonds.append((new_a, new_b))
                if res['status_bond_parameters'] == True:
                    new_bond_parameters.append(bond_parameters[i])
                    status_bond_parameters = True
                else:
                    status_bond_parameters = False
        index_head = new_atoms.index(head) if head != None else None
        index_tail = new_atoms.index(tail) if tail != None else None

        # for angles
        status_angles = res["status_angles"]
        if status_angles == True:
            angles = res['angles']
            angle_parameters = res['angle_parameters']
            new_angles = []
            new_angle_parameters = []
            for i, ang in enumerate(angles):
                a, b, c = ang 
                atom_a, atom_b, atom_c  = atoms[a], atoms[b], atoms[c]
                atom_type_a, atom_type_b = atom_types[a], atom_types[b]
                if (a in delete_index) or (b in delete_index) or (c in delete_index):
                    pass
                else:
                    new_a = new_atoms.index(atom_a) 
                    new_b = new_atoms.index(atom_b) 
                    new_c = new_atoms.index(atom_c) 
                    new_angles.append((new_a, new_b, new_c))
                    new_angle_parameters.append(angle_parameters[i])
        else:
            new_angles, _ = DeduceTopolgy(new_bonds)
            new_angle_parameters = None

        # for dihs
        status_dihs = res['status_dihs']
        if status_dihs == True:
            dihs = res['dihs']
            dih_parameters = res['dih_parameters']
            new_dihs = []
            new_dih_parameters = []
            for i, dih in enumerate(dihs):
                a, b, c, d = dih
                atom_a, atom_b, atom_c, atom_d  = atoms[a], atoms[b], atoms[c], atoms[d]
                if (a in delete_index) or (b in delete_index) or (c in delete_index) or (d in delete_index):
                    pass
                else:
                    new_a = new_atoms.index(atom_a) 
                    new_b = new_atoms.index(atom_b) 
                    new_c = new_atoms.index(atom_c) 
                    new_d = new_atoms.index(atom_d) 
                    new_dihs.append((new_a, new_b, new_c, new_d))
                    new_dih_parameters.append(dih_parameters[i])
        else:
            _, new_dihs = DeduceTopolgy(new_bonds)
            new_dihs = np.array(new_dihs, dtype=np.int)
            new_dih_parameters = None

        _residue = residue_type(name, new_atoms, new_atom_charges, status_atom_charges, new_atom_types, \
                            new_bonds, new_bond_parameters, status_bond_parameters, \
                            new_angles, new_angle_parameters, status_angles, 
                            new_dihs,   new_dih_parameters, status_dihs, \
                                    index_head, index_tail)
        return _residue 
        
    def __init__(self, filename, needed=[]):
        tree = cElementTree.ElementTree(file=filename)
        root = tree.getroot()
        configuration = root
        self.nodes = {}
        for e in configuration:
            if (len(needed) != 0) and (not e.tag in needed):
                continue
            self.nodes[e.tag] = {} 

            # add default prop
            self.nodes[e.tag]['status_angles'] = False 
            self.nodes[e.tag]['status_dihs'] = False 
            for c in e:
                if c.tag == 'atoms':
                    _atoms = read_csv(StringIO(c.text), delim_whitespace=True,  header=None).values
                    _atoms[:, 0] -= _atoms[:, 0].min()
                    self.nodes[e.tag][c.tag] = _atoms 
                elif c.tag == 'bonds':
                    bds = [(_.strip()).split() for _ in c.text.strip().split("\n")]
                    bonds = []

                    # check the bonds form
                    bonds_column_status = True 
                    for bd in bds:
                        if len(bd) != 5:
                            bonds_column_status = False 
                    if bonds_column_status == True:
                        bds = np.array(bds,dtype=np.float)
                        bonds = (bds[:, :2] - bds[:, :2].min()).astype(np.int)
                        bond_parameters = bds[:, 2:]
                        self.nodes[e.tag]['bonds'] = bonds
                        self.nodes[e.tag]['bond_parameters'] = bond_parameters 
                        self.nodes[e.tag]['status_bond_parameters'] = True
                    else:
                        self.nodes[e.tag]['status_bond_parameters'] = False 
                        self.nodes[e.tag]['bond_parameters'] = None
                        indexs = np.asarray(list(chain.from_iterable(bds))).astype(np.int)
                        for bd in bds:
                            for b in bd[1:]:
                                p = int(bd[0])
                                q = int(b)
                                min_bd = np.array([p, q]).min()
                                max_bd = np.array([p, q]).max()
                                _bond = (min_bd, max_bd)
                                bonds.append(_bond)
                        bonds = np.unique(np.asarray(bonds).astype(np.int) - indexs.min(), axis=0)
                        self.nodes[e.tag]['bonds'] = bonds
                elif c.tag == 'angles':
                    angle_info = np.array([(_.strip()).split() for _ in c.text.strip().split("\n")])
                    angles = angle_info[:, :3].astype(np.int)
                    angles = angles - int(angles.min())
                    angle_parameters = angle_info[:, 3:]
                    self.nodes[e.tag]['angles'] = angles
                    self.nodes[e.tag]['angle_parameters'] = angle_parameters 
                    self.nodes[e.tag]['status_angles'] = True
                elif c.tag == 'dihs':
                    dih_info = [(_.strip()).split() for _ in c.text.strip().split("\n")]
                    dihs = np.array([_[:4] for _ in dih_info], dtype=np.int)
                    dihs = dihs - int(dihs.min()) 
                    dih_parameters = [_[4:] for _ in dih_info] 
                    self.nodes[e.tag]['dihs'] = dihs 
                    self.nodes[e.tag]['dih_parameters'] = dih_parameters 
                    self.nodes[e.tag]['status_dihs'] = True


def DeduceTopolgy(_bonds):
    '''
    deduce all angles, dihdedrals from input bonds info.
    '''
    nbonds = len(_bonds)
    _angles = []
    # angles
    for i in range(nbonds):
        for j in range(i+1, nbonds):
            a = _bonds[i]
            b = _bonds[j]
            center = list(set(a).intersection(set(b)))
            if len(center) == 0:
                continue
            else:
                diff = list(set(b).difference(set(a))) + list(set(a).difference(set(b)))
                _max = diff[0] if diff[0] > diff[1] else diff[1]
                _min = diff[0] if diff[0] < diff[1] else diff[1]
                angle = [_min, center[0], _max]
                _angles.append(angle)

    # dihs
    nangles = len(_angles)
    _dihs = []
    for i in range(nangles):
        for j in range(i+1, nangles):
            a = _angles[i] 
            b = _angles[j]
            if a[1] == b[1]: 
                continue
            else:
                center = list(set(a).intersection(set(b)))
                if len(center) != 2:
                    continue
                else:
                    diff = list(set(b).difference(set(a))) + list(set(a).difference(set(b)))
                    _max = diff[0] if diff[0] > diff[1] else diff[1]
                    _min = diff[0] if diff[0] < diff[1] else diff[1]
                    dih1 = [_min, center[0], center[1], _max]
                    dih2 = [_max, center[0], center[1], _min]
                    if dih1[:3] in [a, b] or (dih1[:3])[::-1] in [a, b]:
                        dih = dih1.copy()
                    else:
                        dih = dih2.copy()

                    _dihs.append(dih)
    return _angles, _dihs

def DefineForceField():
    '''
    define force field
    '''
    _F = force_field('jxm')
    _F.atomtypes = _F.cls_atomtypes()
    _F.atomtypes.name = {}
    _F.atomtypes.name['jxm_001'] = {'bond_type':'CT', 'mass':'12.0110', 'charge':-0.180,  'sigma':'3.50000E-01', 'epsilon':'2.76144E-01'}
    _F.atomtypes.name['jxm_002'] = {'bond_type':'CT', 'mass':'12.0110', 'charge':-0.120,  'sigma':'3.50000E-01', 'epsilon':'2.76144E-01'}
    _F.atomtypes.name['jxm_003'] = {'bond_type':'CT', 'mass':'12.0110', 'charge':-0.060,  'sigma':'3.50000E-01', 'epsilon':'2.76144E-01'}
    _F.atomtypes.name['jxm_004'] = {'bond_type':'HC', 'mass':'1.0080',  'charge':0.060,   'sigma':'2.50000E-01', 'epsilon':'1.25520E-01'}
    _F.atomtypes.name['jxm_005'] = {'bond_type':'OH', 'mass':'15.9900', 'charge':-0.6939, 'sigma':'3.12000E-01', 'epsilon':'7.11280E-01'}
    _F.atomtypes.name['jxm_006'] = {'bond_type':'HO', 'mass':'1.0080',  'charge':0.0060,  'sigma':'2.50000E-01', 'epsilon':'1.25520E-01'}
    _F.atomtypes.name['jxm_007'] = {'bond_type':'O2', 'mass':'15.9900', 'charge':-0.4491, 'sigma':'2.50000E-01', 'epsilon':'1.25520E-01'} # O atom of CO in COOH 
    _F.atomtypes.name['jxm_008'] = {'bond_type':'CO', 'mass':'15.9900', 'charge':0.524,   'sigma':'3.50000E-01', 'epsilon':'2.76144E-01'} # O atom of CO in COOH 

    _F.bondtypes = _F.cls_bondtypes()
    _F.bondtypes.name = {}
    _F.bondtypes.name['CT-CT'] = {'func':1, 'parameter':'0.1529 224262.400'}
    _F.bondtypes.name['CT-HC'] = {'func':1, 'parameter':'0.1090 284512.000'}
    _F.bondtypes.name['OH-HO'] = {'func':1, 'parameter':'0.0945 462750.400'}
    _F.bondtypes.name['OH-CT'] = {'func':1, 'parameter':'0.1410 267776.000'}
    _F.bondtypes.name['O2-CO'] = {'func':1, 'parameter':'0.1229 476976.000'}

    _F.angletypes = _F.cls_angletypes()
    _F.angletypes.name = {}
    _F.angletypes.name['CT-CT-HC'] = {'func':1, 'parameter':'110.700    313.800'}
    _F.angletypes.name['HC-CT-HC'] = {'func':1, 'parameter':'107.800    276.144'}
    _F.angletypes.name['CT-CT-CT'] = {'func':1, 'parameter':'112.700    488.273'}
    _F.angletypes.name['OH-CT-CT'] = {'func':1, 'parameter':'109.500    418.400'}
    _F.angletypes.name['CT-OH-HO'] = {'func':1, 'parameter':'108.500    460.240'}
    _F.angletypes.name['OH-CT-HC'] = {'func':1, 'parameter':'109.500    292.880'}
    _F.angletypes.name['O2-CO-OH'] = {'func':1, 'parameter':'121.000    669.440'}
    _F.angletypes.name['O2-CO-CT'] = {'func':1, 'parameter':'120.400    669.440'}

    _F.dihedraltypes = _F.cls_dihedraltypes()
    _F.dihedraltypes.name = {} 
    _F.dihedraltypes.name['HC-CT-CT-HC'] = {'func':3, 'parameter':'0.628   1.883   0.000  -2.510  -0.000   0.000'}
    _F.dihedraltypes.name['CT-CT-CT-CT'] = {'func':3, 'parameter':'2.301  -1.464   0.837  -1.674  -0.000   0.000'}
    _F.dihedraltypes.name['HC-CT-CT-CT'] = {'func':3, 'parameter':'0.628   1.883   0.000  -2.510  -0.000   0.000'}
    _F.dihedraltypes.name['OH-CT-CT-CT'] = {'func':3, 'parameter':'-3.247   3.247   0.000  -0.000  -0.000   0.000'}
    _F.dihedraltypes.name['OH-CT-CT-CT'] = {'func':3, 'parameter':'-3.247   3.247   0.000  -0.000  -0.000   0.000'}
    _F.dihedraltypes.name['OH-CT-CT-HC'] = {'func':3, 'parameter':'0.979   2.937   0.000  -3.916  -0.000   0.000'}
    _F.dihedraltypes.name['HC-CT-OH-HO'] = {'func':3, 'parameter':'0.736   2.209   0.000  -2.946  -0.000   0.000'}
    _F.dihedraltypes.name['CT-CT-OH-HO'] = {'func':3, 'parameter':'-0.444   3.833   0.728  -4.117  -0.000   0.000'}
    _F.dihedraltypes.name['OH-CT-CT-CT'] = {'func':3, 'parameter':'-0.444   3.833   0.728  -4.117  -0.000   0.000'}
    _F.dihedraltypes.name['CT-CT-CO-O2'] = {'func':3, 'parameter':'0.000   0.000   0.000  -0.000  -0.000   0.000'}
    _F.dihedraltypes.name['HC-CT-CO-O2'] = {'func':3, 'parameter':'0.000   0.000   0.000  -0.000  -0.000   0.000'}
    _F.dihedraltypes.name['HO-OH-CO-O2'] = {'func':3, 'parameter':'23.012   0.000  -23.012  -0.000  -0.000  0.000'}

    return _F

def DefineResidueTypes():
    '''
    define residue type, similar to the rtp file format in GROMACS.
    '''
    _ResidueTypes = {} 

    # add PE/NB merged monomer
    #  xml = hoomd_xml('residuetypes.xml', needed=['PE', 'CH2OH_endo', 'COOH_endo', 'COOH_exo'])
    xml = hoomd_xml('residuetypes.xml')
    #  _ResidueTypes['NBH'] = xml.get_res('NB', None, 'C2', 'H1C1', 'H1C2', 'jxm_003', 'jxm_003')
    #  _ResidueTypes['NBM'] = xml.get_res('NB', 'C1', 'C2', 'H1C1', 'H1C2', 'jxm_003', 'jxm_003')
    #  _ResidueTypes['NBT'] = xml.get_res('NB', 'C1', None, 'H1C1', 'H1C2', 'jxm_003', 'jxm_003')

    # add PE
    _ResidueTypes['PEH'] = xml.get_res('PE',  None, 'C01', 'H04', 'H07', 'jxm_002', 'jxm_002', -0.16, -0.16)
    _ResidueTypes['PEM'] = xml.get_res('PE',  'C00','C01', 'H04', 'H07', 'jxm_002', 'jxm_002', -0.16, -0.16)
    _ResidueTypes['PET'] = xml.get_res('PE',  'C00', None, 'H04', 'H07', 'jxm_002', 'jxm_002', -0.16, -0.16)

    # add NB 
    _ResidueTypes['NBH'] = xml.get_res('NB',  None, 'C01', 'H07', 'H09', 'jxm_003', 'jxm_003', -0.0877, -0.0879)
    _ResidueTypes['NBM'] = xml.get_res('NB',  'C00', 'C01', 'H07', 'H09', 'jxm_003', 'jxm_003', -0.0877, -0.0879)
    _ResidueTypes['NBT'] = xml.get_res('NB',  'C00', None, 'H07', 'H09', 'jxm_003', 'jxm_003', -0.0877, -0.0879)

    # add CH2OH_endo
    _ResidueTypes['ENH'] = xml.get_res('CH2OH_endo',  None, 'C06', 'H0G', 'H0J', 'jxm_003', 'jxm_003', -0.0873, -0.0892)
    _ResidueTypes['ENM'] = xml.get_res('CH2OH_endo',  'C05', 'C06', 'H0G', 'H0J', 'jxm_003', 'jxm_003', -0.0873, -0.0892)
    _ResidueTypes['ENT'] = xml.get_res('CH2OH_endo',  'C05', None, 'H0G', 'H0J', 'jxm_003', 'jxm_003', -0.0873, -0.0892)

    # add CH2OH_exo
    _ResidueTypes['EOH'] = xml.get_res('CH2OH_exo',  None, 'C07', 'H0G', 'H0J', 'jxm_003', 'jxm_003', -0.0893, -0.0875)
    _ResidueTypes['EOM'] = xml.get_res('CH2OH_exo',  'C06', 'C07', 'H0G', 'H0J', 'jxm_003', 'jxm_003', -0.0893, -0.0875)
    _ResidueTypes['EOT'] = xml.get_res('CH2OH_exo',  'C06', None, 'H0G', 'H0J', 'jxm_003', 'jxm_003', -0.0893, -0.0875)

    # add COOH_endo
    _ResidueTypes['COH'] = xml.get_res('COOH_endo',  None, 'C07', 'H0G', 'H0H', 'jxm_003', 'jxm_003', -0.0788, -0.0814)
    _ResidueTypes['COM'] = xml.get_res('COOH_endo',  'C06', 'C07', 'H0G', 'H0H', 'jxm_003', 'jxm_003', -0.0788, -0.0814)
    _ResidueTypes['COT'] = xml.get_res('COOH_endo',  'C06', None, 'H0G', 'H0H', 'jxm_003', 'jxm_003', -0.0788, -0.0814)

    # add COOH_exo
    _ResidueTypes['CXH'] = xml.get_res('COOH_exo',  None, 'C08', 'H0G', 'H0H', 'jxm_003', 'jxm_003', -0.0877, -0.0878)
    _ResidueTypes['CXM'] = xml.get_res('COOH_exo',  'C07', 'C08', 'H0G', 'H0H', 'jxm_003', 'jxm_003', -0.0877, -0.0878)
    _ResidueTypes['CXT'] = xml.get_res('COOH_exo',  'C07', None, 'H0G', 'H0H', 'jxm_003', 'jxm_003', -0.0877, -0.0878)

    # add OH
    _ResidueTypes['OHH'] = xml.get_res('NBOH',  None, 'C05', 'H0D', 'H0G', 'jxm_003', 'jxm_003', -0.0873, -0.0714)
    _ResidueTypes['OHM'] = xml.get_res('NBOH',  'C04', 'C05', 'H0D', 'H0G', 'jxm_003', 'jxm_003', -0.0873, -0.0714)
    _ResidueTypes['OHT'] = xml.get_res('NBOH',  'C04', None, 'H0D', 'H0G', 'jxm_003', 'jxm_003', -0.0873, -0.0714)

    print("Add ResidueTypes Done --------------------------------------------------------")

    return _ResidueTypes

def DefineResidueSeq():
    '''
    define user residue sequence and number here
    '''
    #  resname = ["ETH", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETT"]
    #  nres  =   [1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 9, 1, 9, 1, 9, 1, 9,1, 1,1,1,1,1,1, 1]
    #  resname = ['ETH', 'NB', 'ETT']
    #  nres = [1, 1, 1]

    try:
        info = sys.argv[1]
        with open(info, 'r') as fp:
            resname = []
            nres = []
            for line in fp:
                sp = line.split()
                if len(sp) == 2:
                    resname.append(sp[0])
                    nres.append(int(sp[1]))
    except:
        print("ERROR! Residue Input Info Is Wrong!")
        print("PLEASE ENSURE THE RESIDUE INPUT FILE IS AVILABLE AND THE FORMAT IS CORRENT")
        print("EXAMPLE:")
        print("  ETH   1")
        print("  ETM   1")
        print("  NB    1")
        print("  ETT   1")
        sys.exit()

    _AllResname = [[resname[i]] * nres[i] for i in range(len(nres))]
    _AllResname = [item for sub in _AllResname for item in sub]
    
    _natoms = np.array([ResidueTypes[_AllResname[idx]].natoms for idx in range(len(_AllResname))]).astype(np.int).sum()

    # dump info
    print("SYSTEM INFO: >>>>>>>>")
    print("    Total Residues: %d "%(len(_AllResname)))
    print("    Total Residues Types: %d "%(len(set(_AllResname))))
    print("    Total Atoms: %d "%(_natoms))
    print("----------------------------------------------------------------")

    return _AllResname

def BuildTopology():
    '''
    build topology based on 
    1. Force field
    2. Residue type
    3. User input residue sequence info
    '''
    global S
    S = System()

    from datetime import datetime
    # make top - headers 
    o = open("topol.top", 'w')
    o.write("\n")
    o.write("; GENERATED at  %s\n"%(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    o.write("; Author: Jia Xiang Meng @ Jilin University\n")
    o.write(";\n")
    o.write("\n")
    o.write("[ defaults  ]\n")
    o.write("; nbfunc    comb-rule   gen-pairs   fudgeLJ fudgeQQ\n")
    o.write("1       3       yes     0.5 0.5\n")
    o.write(";\n")

    # make top - [ atomtypes ]
    o.write("\n")
    o.write("[ atomtypes ] \n")
    for k, v in F.atomtypes.name.items():
        name = k
        bond_type = v['bond_type']
        mass      = v['mass']
        charge    = v['charge']
        sigma     = v['sigma']
        epsilon   = v['epsilon']
        o.write("  %s  %s  %s  %s  %s  %s  %s\n"%(name, bond_type, mass, charge, 'A', sigma, epsilon))

    # make top - [ moleculetype ]
    o.write("\n")
    o.write("\n")
    o.write("[ moleculetype ]\n")
    o.write("; Name               nrexcl\n")
    o.write("NB                   3\n")

    # write [ atoms ]
    o.write("[ atoms ]\n")
    o.write(";   nr       type  resnr residue  atom   cgnr     charge       mass\n")
    counter = 0
    for i, res in enumerate(AllResname):
        resnr = i+1
        R = ResidueTypes[res]
        residue = res 
        natoms = R.natoms
        o.write("; residue  %d %s    total charges: %.4f\n"%(resnr, res, S.ncharges))

        # for charges
        if R.status_atom_charges == True:
            charges = R.atom_charges

        for j in range(natoms):
            nr =  counter + 1
            type = R.atom_types[j]
            ff = F.atomtypes.name[type]
            mass = ff['mass']
            atom = R.atoms[j]
            cgnr = i+1
            if R.status_atom_charges == True:
                charge = charges[j]
            else:
                charge = ff['charge']
            S.ncharges += charge
            o.write("    %d  %s  %d  %s  %s  %d  %.4f  %s\n"%(nr, type, resnr, residue, atom, cgnr, charge, mass))

            bond_type = ff['bond_type']

            counter += 1

    # write [ bonds ]
    o.write("\n")
    o.write("\n")
    o.write("[ bonds ]\n")
    o.write(";  ai    aj funct            c0            c1            c2            c3\n")
    counter = 0
    former_tail = None
    for i, res in enumerate(AllResname):
        resnr = i+1
        R = ResidueTypes[res]
        bonds = R.bonds
        R.resnr = resnr
        o.write("; residue  %d %s\n"%(resnr, res))

        if R.status_bond_parameters == True:
            bond_parameters = R.bond_parameters
            for bd, bp in zip(bonds, bond_parameters):
                ai = bd[0] + counter + 1
                aj = bd[1] + counter + 1
                func = bp[0]
                angle = bp[1]
                kkk = bp[2]
                o.write("   %d   %d  %d  %.6f  %.6f\n"%(ai, aj, func, angle, kkk))
        else: 
            for bd in bonds:
                ai = bd[0]
                aj = bd[1]
                ai_type = R.atom_types[ai]
                aj_type = R.atom_types[aj]
                ai_bond_type = F.atomtypes.name[ai_type]['bond_type']
                aj_bond_type = F.atomtypes.name[aj_type]['bond_type']
                bond_type = ai_bond_type + '-' + aj_bond_type
                r_bond_type = aj_bond_type + '-' + ai_bond_type

                # write bond
                aii = ai + counter + 1
                ajj = aj + counter + 1
                if bond_type in F.bondtypes.name:
                    func = F.bondtypes.name[bond_type]['func']
                    parameter = F.bondtypes.name[bond_type]['parameter']
                    o.write("   %d   %d  %d  %s\n"%(aii, ajj, func, parameter))
                elif r_bond_type in F.bondtypes.name:
                    func = F.bondtypes.name[r_bond_type]['func']
                    parameter = F.bondtypes.name[r_bond_type]['parameter']
                    o.write("   %d   %d  %d  %s\n"%(aii, ajj, func, parameter))
                else:
                    print("ERROR! %s (or %s)not found in force field"%(bond_type, r_bond_type))
                    print("ERROR! %s %d-%d"%(res, ai, aj))
                    sys.exit()
        
        # add inter-residue linking bonds
        if R.head != None:
            ai = former_R.tail 
            aj = R.head
            ai_type = former_R.atom_types[ai]
            aj_type = R.atom_types[aj]
            ai_bond_type = F.atomtypes.name[ai_type]['bond_type']
            aj_bond_type = F.atomtypes.name[aj_type]['bond_type']
            bond_type = ai_bond_type + '-' + aj_bond_type
            r_bond_type = aj_bond_type + '-' + ai_bond_type

            # write bond
            aii = ai + former_R.counter + 1
            ajj = aj + counter + 1
            if bond_type in F.bondtypes.name:
                func = F.bondtypes.name[bond_type]['func']
                parameter = F.bondtypes.name[bond_type]['parameter']
                o.write("   %d   %d  %d  %s  "%(aii, ajj, func, parameter))
            elif r_bond_type in F.bondtypes.name:
                func = F.bondtypes.name[r_bond_type]['func']
                parameter = F.bondtypes.name[r_bond_type]['parameter']
                o.write("   %d   %d  %d  %s"%(aii, ajj, func, parameter))
            else:
                print("ERROR! %s (or %s)not found in force field"%(bond_type, r_bond_type))
                sys.exit()
            o.write(" ; >>>  inter-residue linking bond\n")

        R.counter = counter
        if R.tail != None:
            former_R = R

        # end
        counter += R.natoms

    # write [ angles ]
    o.write("\n\n")
    o.write("[ angles ]\n")
    o.write(";  ai    aj    ak funct            c0            c1            c2            c3\n")
    counter = 0
    for i, res in enumerate(AllResname):
        R = ResidueTypes[res]
        angles = R.angles
        resnr = i+1
        o.write("; residue  %d %s\n"%(resnr, res))
        if R.status_angles == True:
            angle_parameters = R.angle_parameters
            for angle, ap in zip(angles, angle_parameters):
                aii = angle[0] + counter + 1
                ajj = angle[1] + counter + 1
                akk = angle[2] + counter + 1
                func = int(ap[0])
                theta = float(ap[1])
                kkk = float(ap[2])
                o.write("   %d   %d  %d  %d  %.6f  %.6f\n"%(aii, ajj, akk, func, theta, kkk))
                #  print("Write angls:=------------------------")
        else:
            for angle in angles:
                ai = angle[0]
                aj = angle[1]
                ak = angle[2]
                ai_type = R.atom_types[ai]
                aj_type = R.atom_types[aj]
                ak_type = R.atom_types[ak]
                angle_atoms = ai_type + '-' + aj_type + '-' + ak_type

                ai_angle_type = F.atomtypes.name[ai_type]['bond_type']
                aj_angle_type = F.atomtypes.name[aj_type]['bond_type']
                ak_angle_type = F.atomtypes.name[ak_type]['bond_type']
                angle_type = ai_angle_type + '-' + aj_angle_type + '-' + ak_angle_type
                r_angle_type = ak_angle_type + '-' + aj_angle_type + '-' +  ai_angle_type
                aii = ai + counter + 1
                ajj = aj + counter + 1
                akk = ak + counter + 1
                if angle_type in F.angletypes.name:
                    func = F.angletypes.name[angle_type]['func']
                    parameter = F.angletypes.name[angle_type]['parameter']
                    o.write("   %d   %d  %d  %d  %s\n"%(aii, ajj, akk, func, parameter))
                elif r_angle_type in F.angletypes.name:
                    func = F.angletypes.name[r_angle_type]['func']
                    parameter = F.angletypes.name[r_angle_type]['parameter']
                    o.write("   %d   %d  %d  %d  %s\n"%(aii, ajj, akk, func, parameter))
                else:
                    print("ERROR! %s (or %s)not found in force field"%(angle_type, r_angle_type))
                    sys.exit()

        # add inter-residdue angles
        if R.head != None:
            ai = former_R.tail
            ai_type = former_R.atom_types[ai]
            ai_angle_type = F.atomtypes.name[ai_type]['bond_type']
            ai_bond = former_R.get_linkbond(ai)
            aii = ai + former_R.counter + 1
            former_third = [_[0] if (_.tolist().index(ai))== 1 else _[1] for _ in ai_bond]
            former_third_type = [former_R.atom_types[_] for _ in former_third]
            former_third_angle_type = [F.atomtypes.name[_]['bond_type'] for _ in former_third_type]
                
            aj = R.head
            aj_type = R.atom_types[aj]
            aj_angle_type = F.atomtypes.name[aj_type]['bond_type']
            aj_bond = R.get_linkbond(aj)
            ajj = aj + counter + 1
            third = [_[0] if (_.tolist().index(aj))== 1 else _[1] for _ in aj_bond]
            third_type = [R.atom_types[_] for _ in third]
            third_angle_type = [F.atomtypes.name[_]['bond_type'] for _ in third_type]
            
            for j, p in enumerate(former_third_angle_type):
                angle_type = p + '-' + ai_angle_type + '-' + aj_angle_type
                r_angle_type = aj_angle_type + '-' + ai_angle_type + '-' + p
                akk = former_R.counter + former_third[j] + 1
                if angle_type in F.angletypes.name:
                    func = F.angletypes.name[angle_type]['func']
                    parameter = F.angletypes.name[angle_type]['parameter']
                    o.write("   %d   %d  %d  %d  %s "%(akk, aii, ajj, func, parameter))
                elif r_angle_type in F.angletypes.name:
                    func = F.angletypes.name[r_angle_type]['func']
                    parameter = F.angletypes.name[r_angle_type]['parameter']
                    o.write("   %d   %d  %d  %d  %s "%(akk, aii, ajj, func, parameter))
                else:
                    print("ERROR! %s (or %s)not found in force field"%(angle_type, r_angle_type))
                    print("ERROR! %d %d %d"%(akk, aii, ajj))
                    sys.exit()
                o.write(" ; >>>  inter-residue linking angle\n")

            for j, p in enumerate(third_angle_type):
                angle_type = ai_angle_type + '-' + aj_angle_type + '-' + p
                r_angle_type = p + '-' + aj_angle_type + '-' + ai_angle_type
                akk = counter + third[j] + 1
                if angle_type in F.angletypes.name:
                    func = F.angletypes.name[angle_type]['func']
                    parameter = F.angletypes.name[angle_type]['parameter']
                    o.write("   %d   %d  %d  %d  %s "%(aii, ajj, akk, func, parameter))
                elif r_angle_type in F.angletypes.name:
                    func = F.angletypes.name[r_angle_type]['func']
                    parameter = F.angletypes.name[r_angle_type]['parameter']
                    o.write("   %d   %d  %d  %d  %s "%(aii, ajj, akk, func, parameter))
                else:
                    print("ERROR! %s (or %s)not found in force field"%(angle_type, r_angle_type))
                    sys.exit()
                o.write(" ; >>>  inter-residue linking angle\n")
                
        R.counter = counter
        if R.tail != None:
            former_R = R

        # end
        counter += R.natoms

    # write [ dihedrals ]
    o.write("\n\n")
    o.write("[ dihedrals ]\n")
    o.write(";  ai    aj    ak    al      funct            c0            c1            c2            c3            c4            c5\n")
    counter = 0
    for i, res in enumerate(AllResname):
        R = ResidueTypes[res]
        dihs = R.dihs
        resnr = i+1
        o.write("; residue  %d %s\n"%(resnr, res))
        if R.status_dihs == True:
            dih_parameters = R.dih_parameters
            for dih, dp in zip(dihs, dih_parameters):
                aii = dih[0] + counter + 1
                ajj = dih[1] + counter + 1
                akk = dih[2] + counter + 1
                aee = dih[3] + counter + 1
                func = int(dp[0])
                if func == 4:
                    theta, kkk, num = float(dp[1]), float(dp[2]), int(dp[3])
                    o.write("   %d %d  %d  %d  %d %.6f %.6f %d  ; improper dihs\n"%(aii, ajj, akk, aee, func, theta, kkk, num))
                if func == 3:
                    r1, r2, r3, r4, r5, r6 = np.array(dp, dtype=np.float)[1:]
                    o.write("   %d %d  %d  %d  %d %.6f %.6f %.6f %.6f %.6f %.6f\n"%(aii, ajj, akk, aee, func, r1, r2, r3, r4, r5, r6))
        else:
            # add improper  to stero
            if res in ['EOH', 'EOM', 'EOT']:
                # grasp atoms
                index_C01 = R.atoms.index('C01') + counter + 1
                index_C02 = R.atoms.index('C02') + counter + 1
                index_C04 = R.atoms.index('C04') + counter + 1
                index_C05 = R.atoms.index('C05') + counter + 1
                index_C07 = R.atoms.index('C07') + counter + 1
                index_C08 = R.atoms.index('C08') + counter + 1
                index_H03 = R.atoms.index('H03') + counter + 1
                index_H0K = R.atoms.index('H0K') + counter + 1
                dih_3 = (index_C08, index_C02, index_C04, index_C01)
                dih_4 = (index_H0K, index_C08, index_C02, index_H03)
                parameter = '76.0     1000'
                o.write("   %d   %d   %d  %d  %d %s ; ==================================== stero control\n"%(dih_3[0], dih_3[1], dih_3[2], dih_3[3], 2, parameter))
                parameter = '82.0     1000'
                o.write("   %d   %d   %d  %d  %d %s ; ==================================== stero control\n"%(dih_4[0], dih_4[1], dih_4[2], dih_4[3], 2, parameter))

            for dih in dihs:
                ai = dih[0]
                aj = dih[1]
                ak = dih[2]
                al = dih[3]


                index_dih = (ai, aj, ak, al)
                atoms_dih = (R.atoms[ai], R.atoms[aj], R.atoms[ak], R.atoms[al])
                ai_type = R.atom_types[ai]
                aj_type = R.atom_types[aj]
                ak_type = R.atom_types[ak]
                al_type = R.atom_types[al]
                ai_dih_type = F.atomtypes.name[ai_type]['bond_type']
                aj_dih_type = F.atomtypes.name[aj_type]['bond_type']
                ak_dih_type = F.atomtypes.name[ak_type]['bond_type']
                al_dih_type = F.atomtypes.name[al_type]['bond_type']
                aii = ai + counter + 1
                ajj = aj + counter + 1
                akk = ak + counter + 1
                all = al + counter + 1
                dih_type = ai_dih_type + '-' + aj_dih_type + '-' + ak_dih_type + '-' + al_dih_type
                r_dih_type = al_dih_type + '-' + ak_dih_type + '-' + aj_dih_type + '-' +  ai_dih_type 
                if res in ['EOH', 'EOM', 'EOT']:
                    # grasp atoms
                    index_C01 = R.atoms.index('C01')
                    index_C02 = R.atoms.index('C02')
                    index_C04 = R.atoms.index('C04')
                    index_C05 = R.atoms.index('C05')
                    index_C07 = R.atoms.index('C07')
                    index_C08 = R.atoms.index('C08')
                    dih_1 = (index_C07, index_C08, index_C02, index_C01)
                    dih_2 = (index_C05, index_C04, index_C02, index_C01)
                    if (index_dih == dih_1) or (index_dih[::-1] == dih):
                        break
                        func = 2
                        parameter = '18.0     1000'
                        o.write("   %d   %d   %d  %d  %d %s ; ==================================== stero control\n"%(aii, ajj, akk, all, func, parameter))
                    elif (index_dih == dih_2) or (index_dih[::-1] == dih_2):
                        break
                        func = 2
                        parameter = '56.0     1000'
                        o.write("   %d   %d   %d  %d  %d %s ; ==================================== stero control\n"%(aii, ajj, akk, all, func, parameter))
                    else:
                        if dih_type in F.dihedraltypes.name:
                            func = F.dihedraltypes.name[dih_type]['func']
                            parameter = F.dihedraltypes.name[dih_type]['parameter']
                            o.write("   %d   %d   %d  %d  %d %s\n"%(aii, ajj, akk, all, func, parameter))
                        elif r_dih_type in F.dihedraltypes.name:
                            func = F.dihedraltypes.name[r_dih_type]['func']
                            parameter = F.dihedraltypes.name[r_dih_type]['parameter']
                            o.write("   %d %d   %d  %d  %d %s\n"%(aii, ajj, akk, all, func, parameter))
                        else:
                            print("ERROR! %s (or %s)not found in force field"%(dih_type, r_dih_type))
                            print("ERROR! Dih-index:%d %d %d %d"%(ai, aj, ak, al))
                            sys.exit()

                else:
                    if dih_type in F.dihedraltypes.name:
                        func = F.dihedraltypes.name[dih_type]['func']
                        parameter = F.dihedraltypes.name[dih_type]['parameter']
                        o.write("   %d   %d   %d  %d  %d %s\n"%(aii, ajj, akk, all, func, parameter))
                    elif r_dih_type in F.dihedraltypes.name:
                        func = F.dihedraltypes.name[r_dih_type]['func']
                        parameter = F.dihedraltypes.name[r_dih_type]['parameter']
                        o.write("   %d %d   %d  %d  %d %s\n"%(aii, ajj, akk, all, func, parameter))
                    else:
                        print("ERROR! %s (or %s)not found in force field"%(dih_type, r_dih_type))
                        print("ERROR! Dih-index:%d %d %d %d"%(ai, aj, ak, al))
                        sys.exit()
                
        if R.head != None:
            ai = former_R.tail
            ai_type = former_R.atom_types[ai]
            ai_angle_type = F.atomtypes.name[ai_type]['bond_type']
            ai_bond = former_R.get_linkbond(ai)
            aii = ai + former_R.counter + 1
            former_third = [_[0] if (_.tolist().index(ai))== 1 else _[1] for _ in ai_bond]
            former_third_type = [former_R.atom_types[_] for _ in former_third]
            former_third_angle_type = [F.atomtypes.name[_]['bond_type'] for _ in former_third_type]
                
            aj = R.head
            aj_type = R.atom_types[aj]
            aj_angle_type = F.atomtypes.name[aj_type]['bond_type']
            aj_bond = R.get_linkbond(aj)
            ajj = aj + counter + 1
            third = [_[0] if (_.tolist().index(aj))== 1 else _[1] for _ in aj_bond]
            third_type = [R.atom_types[_] for _ in third]
            third_angle_type = [F.atomtypes.name[_]['bond_type'] for _ in third_type]

            for j, p in enumerate(former_third_angle_type):
                for k, q in enumerate(third_angle_type):
                    dih_type = p + '-' + ai_angle_type + '-' + aj_angle_type + '-' + q
                    r_dih_type = q + '-' + aj_angle_type + '-' + ai_angle_type + '-' + p
                    akk = former_R.counter + former_third[j] + 1
                    all = counter + third[k] + 1
                    if dih_type in F.dihedraltypes.name:
                        func = F.dihedraltypes.name[dih_type]['func']
                        parameter = F.dihedraltypes.name[dih_type]['parameter']
                        o.write("   %d   %d   %d  %d  %d %s"%(akk, aii, ajj, all, func, parameter))
                    elif r_dih_type in F.dihedraltypes.name:
                        func = F.dihedraltypes.name[r_dih_type]['func']
                        parameter = F.dihedraltypes.name[r_dih_type]['parameter']
                        o.write("   %d %d   %d  %d  %d %s"%(akk, aii, ajj, all, func, parameter))
                    else:
                        print("ERROR! %s (or %s)not found in force field"%(dih_type, r_dih_type))
                        sys.exit()
                    o.write(" ; >>>  inter-residue linking dihedral\n")

        R.counter = counter
        if R.tail != None:
            former_R = R

        # end
        counter += R.natoms

    # write info tails
    o.write("\n\n")
    o.write("[ system  ]\n")
    o.write("; Name\n")
    o.write("Polymer\n")
    o.write("\n")
    o.write("[ molecules  ]\n")
    o.write("; Compound        #mols\n")
    o.write("NB        1\n")

# main module
F = DefineForceField()
ResidueTypes = DefineResidueTypes()
AllResname = DefineResidueSeq()
BuildTopology()
