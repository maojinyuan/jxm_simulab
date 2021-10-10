class Graph(object):
    def __init__(self):
        self.neighbors = {}

    def add_vertex(self, v):
        if v not in self.neighbors:
            self.neighbors[v] = []

    def add_edge(self, u, v):
        self.neighbors[u].append(v)
        # if u == v, do not connect u to itself twice
        # if u != v: # Unconmment if unidirect bond hash was built.
        # self.neighbors[v].append(u)

    def vertices(self):
        return list(self.neighbors.keys())

    def vertex_neighbors(self, v):
        return self.neighbors[v]


def is_cycl(g: Graph) -> bool:
    q = []
    v = g.vertices()
    # initially all vertices are unexplored
    layer = {v: -1 for v in v}
    for v in v:
        # v has already been explored; move on
        if layer[v] != -1:
            continue
        # take v as a starting vertex
        layer[v] = 0
        q.append(v)
        # as long as q is not empty
        while len(q) > 0:
            # get the next vertex u of q that must be looked at
            u = q.pop(0)
            c = g.vertex_neighbors(u)
            for z in c:
                # if z is being found for the first time
                if layer[z] == -1:
                    layer[z] = layer[u] + 1
                    q.append(z)
                elif layer[z] >= layer[u]:
                    return True
    return False


def ggm(bond_hash, molecule):  # Dual bond_hash
    mol_graph = Graph()
    for atom in molecule:
        mol_graph.add_vertex(atom)
    for atom in molecule:
        for btom in bond_hash[atom]:
            mol_graph.add_edge(atom, btom)
    return mol_graph
