import csv

def read_smiles(path):
    # Assumes data is in the form: "smiles, label"
    data = []
    with open(path, 'r') as smiles_file:
        reader = csv.reader(smiles_file)
        _ = next(reader)  # header
        lines = list(reader)

        for smiles, label in lines:
            data.append((smiles, int(label)))
    return data

def read_fg(path):
    # Read a list of functional groups, assumes the data is in the form "smiles"
    fg_list = []
    with open(path, 'r') as fg_file:
        for line in fg_file.readlines():
            fg_list.append(line.strip())
    return fg_list

def write_output(smiles_all, smiles_actives, output_dir, output_prefix):
    # Assumes smiles actives is a strict subset of smiles_all
    actives_set = set(smiles_actives)

    with open('%s/%s_all.csv' % (output_dir, output_prefix), 'w+') as output_file:
        output_file.write('smiles,label\n')

        for smiles in smiles_all:
            if smiles in actives_set:
                label = 1
            else:
                label = 0
            output_file.write('%s,%d\n' % (smiles, label))

    with open('%s/%s_actives.csv' % (output_dir, output_prefix), 'w+') as output_file:
        output_file.write('smiles,label\n')

        for smiles in smiles_actives:
            if smiles in actives_set:
                label = 1
            else:
                label = 0
            output_file.write('%s,%d\n' % (smiles, label))
