def grab_iter_dual(i, bond_hash, mol_used, body_hash=None):
    s = [i]
    r = []
    while s:
        v = s.pop()
        if not mol_used[v]:
            r.append(v)
            mol_used[v] = True
            # for w in bond_hash[v]:
            # s.append(w)
            s.extend(bond_hash[v])
            if not body_hash:
                continue
            for w in body_hash.get(v):
                s.append(w)
                for x in bond_hash[w]:
                    s.append(x)
    return r
