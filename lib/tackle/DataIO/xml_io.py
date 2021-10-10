import numpy as np
from io import StringIO
from xml.etree import cElementTree
from pandas import read_csv

class xml_parser(object):
    """Parse file in xml format
    :param filename: string
    :param needed: list with needed nodes filled, such as 'position', 'bond'
    """

    def __init__(self, filename, needed=[]):
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

        self.nodes['position_unwrap'] = self.nodes['position'] + self.box * self.nodes['image']


class xml_writer(object):
    """Write system data into file in xml format
    :param outname: string  
    :param load: class [xml_parser class type]
    :param xml_type: xml type name, galamost/hoomd

    Note: in development
    """

    def __init__(self, outname, load=None, xml_type=None):
        self.o = open(outname, 'w')
        self.xml_type = 'hoomd' if xml_type is None else xml_type
        if load is not None:
            self.config = load.config
            self.box = load.box
            self.nodes = load.nodes
        else:
            self.nodes = {}

    def write(self):
        if not hasattr(self, 'config'):
            self.config = {
                'time_step': 0,
                'dimensions': 3,
                'natoms': len(self.nodes['position'])
            }

        self.formatter = {
            'position': '{:<18.8f}{:<18.8f}{:<18.8f}\n',
            'bond': '{:<s} {:<d} {:<d}\n',
            'angle': '{:<s} {:<d} {:<d} {:<d}\n',
            'image': '{:<d} {:<d} {:<d}\n',
            'type': '{:<s}\n',
            'body': '{:<d}\n',
            'h_init': '{:<d}\n',
            'h_cris': '{:<d}\n',
            'mass': '{:<.4f}\n',
            'dihedral': '{:<s} {:<d} {:<d} {:<d} {:<d}\n'
        }

        # 'write headers'
        self.o.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        self.o.write('<%s_xml version="1.6">\n' % (self.xml_type))
        self.o.write('<configuration time_step="%s" dimensions="%s" natoms="%s">\n' % (self.config['time_step'], self.config['dimensions'], self.config['natoms']))
        self.o.write('<box lx="%f" ly="%f" lz="%f" xy="0" xz="0" yz="0"/>\n' % (self.box[0], self.box[1], self.box[2]))

        for node, value in self.nodes.items():
            num = len(value)

            # write header
            self.o.write('<%s num="%d">\n' % (node, num))

            # write contents
            for p in value:
                if node in self.formatter:
                    self.o.write(self.formatter[node].format(*p))
                else:
                    print("Warning -> [%s] is not in the current formatter - format as string"%(node))
                    for j in range(len(p)):
                        self.o.write('%s '%(node))
                    self.o.write('\n')

            # write tail
            self.o.write('</%s>\n' % (node))

        # 'write tails'
        self.o.write('</configuration>\n')
        self.o.write('</%s_xml>\n' % (self.xml_type))
        self.o.close()


if __name__ == 'main':
    pass
