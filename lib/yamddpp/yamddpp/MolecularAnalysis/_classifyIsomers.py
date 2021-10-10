def classify_isomers(molecular_types_list, isomer_hash):
    used_hash = {}
    mols = len(molecular_types_list)
    for i in range(mols):
        used_hash[i] = False
    for i in range(mols):
        if used_hash[i]:
            continue
        isomer_hash[i] = [i]
        imol_t = sorted(molecular_types_list[i])  # classify by type
        # imol_b = sorted(molecular_bodies_list[i])
        imol_chara = imol_t  # + imol_b
        for j in range(i + 1, mols):
            jmol_t = sorted(molecular_types_list[j])
            # jmol_b = sorted(molecular_bodies_list[j])
            jmol_chara = jmol_t  # + jmol_b
            if imol_chara == jmol_chara:
                used_hash[j] = True
                isomer_hash[i].append(j)
    return None
