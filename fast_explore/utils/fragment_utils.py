import random
import rdkit.Chem as Chem
from tqdm import tqdm
import pickle as pl
import json
import numpy as np
import re
import selfies as sf

from p_tqdm import p_umap

from mol_explore.models.mol_graph import MolGraph

import pdb

ATOM_MAP_KEY = 'molAtomMapNumber'
ATOM_DUMMY_KEY = 1

def add_attachment(smiles, attach_idx):

    mol_graph = MolGraph(smiles)
    emol = Chem.RWMol(mol_graph.mol)  # Create new editable mol
    new_atom = Chem.Atom('*')
    w_idx = emol.AddAtom(new_atom)  # w_idx is the index of the wildcard atom

    emol.AddBond(w_idx, attach_idx, Chem.rdchem.BondType.SINGLE)
    return Chem.MolToSmiles(emol)

def parse_smiles_attachment(smiles=None, mol=None):
    # Input smiles with attachment atom
    # Output the smiles string without the attachment atom, and the attachment idx
    assert smiles is not None or mol is not None
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
    emol = Chem.RWMol(mol)  # Create an editable mol

    w_idx = -1
    attach_atom = None
    for atom in emol.GetAtoms():
        if atom.GetSymbol() == '*':
            neighbor = atom.GetNeighbors()[0]  # wildcard atom only has 1 neighbor
            neighbor.SetAtomMapNum(ATOM_DUMMY_KEY)  # Set the atom number to mark the atom
            attach_atom = neighbor
            w_idx = atom.GetIdx()

    emol.RemoveAtom(w_idx)
    try:
        Chem.SanitizeMol(emol)
    except:
        return None

    new_smiles = Chem.MolToSmiles(emol)

    try:
        new_mol = Chem.MolFromSmiles(new_smiles)
        attach_idx = -1
        for atom in new_mol.GetAtoms():
            if atom.GetAtomMapNum():
                attach_idx = atom.GetIdx()

        new_smiles_check = Chem.MolToSmiles(new_mol)
    except:
        return None

    if new_smiles != new_smiles_check:
        return None

    output = clean_mapped_smiles(new_smiles), attach_idx

    if output[0] is None:
        return None
    else:
        return output

def rand_select_frag(frag_mols, max_fragment_size=10):
    # Have to account for the additional dummy node
    # Filter out the mol if more than the allotted fragment size
    frag_mols_filtered = [mol for mol in frag_mols if mol.GetNumAtoms() <= (max_fragment_size + 1)]

    if len(frag_mols_filtered) == 0:
        return None

    rand_frag = random.choice(frag_mols_filtered)
    return rand_frag

def clean_fragment_smiles(frag_smiles):
    # The first atom in fragment smiles should be the wildcard, but it may or
    # may not be numbered
    valid_check = re.match('^\[[0-9]*\*\]', frag_smiles)
    if valid_check is None:
        valid_check = re.match('^\*', frag_smiles)

    end_idx = valid_check.end()
    frag_smiles_clean = frag_smiles[end_idx:]
    return frag_smiles_clean

def clean_mapped_wc_smiles(frag_smiles):
    clean_smiles = clean_fragment_smiles(frag_smiles)
    return '*' + clean_smiles

def fragment_molecule(smiles=None, mol=None, max_fragment_size=10):
    assert smiles is not None or mol is not None
    if mol is None:
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            return None
    if mol is None:
        return None

    bonds = mol.GetBonds()
    n_bonds = len(bonds)

    valid_bonds = []  # List of ids of all single bonds
    for bond in bonds:
        bond_type = bond.GetBondType()
        bond_idx = bond.GetIdx()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            valid_bonds.append(bond_idx)

    random.shuffle(valid_bonds)

    # If a valid pair is found, returns a tuple of (base_smiles, fragment_smiles)
    for bond_idx in valid_bonds:
        frag_mol = Chem.rdmolops.FragmentOnBonds(mol=mol, bondIndices=[bond_idx], addDummies=True)
        frag_smiles = Chem.MolToSmiles(frag_mol)

        frag_smiles_tuple = frag_smiles.split('.')
        if len(frag_smiles_tuple) != 2:
            continue

        frag_mol_tuple = [Chem.MolFromSmiles(smiles) for smiles in frag_smiles_tuple]
        rand_frag = rand_select_frag(frag_mol_tuple)

        if rand_frag is not None:
            rand_frag_smiles = Chem.MolToSmiles(rand_frag)

            if rand_frag_smiles == frag_smiles_tuple[0]:
                base_smiles = frag_smiles_tuple[1]
            elif rand_frag_smiles == frag_smiles_tuple[1]:
                base_smiles = frag_smiles_tuple[0]
            else:
                assert False
            return clean_mapped_wc_smiles(base_smiles), clean_mapped_wc_smiles(rand_frag_smiles)
    return None

def clean_mapped_smiles(mapped_smiles):
    groups = re.finditer('\[[a-zA-Z0-9:]+\]', mapped_smiles)
    for match in groups:
        start_idx = match.start()
        end_idx = match.end()
        substr = mapped_smiles[start_idx:end_idx]
        if ':' in substr:
            new_smiles = mapped_smiles[:start_idx] + mapped_smiles[start_idx+1] + mapped_smiles[end_idx:]
            return new_smiles

    return None

def clean_base(base_smiles):
    if base_smiles[0] == '*':
        return base_smiles

    start_idx = base_smiles.find('[')
    end_idx = base_smiles.find(']')

    if start_idx != 0:
        pdb.set_trace()
    base_smiles = '*' + base_smiles[end_idx+1:]

    x = Chem.MolFromSmiles(base_smiles)
    return Chem.MolToSmiles(x)

def gen_selfies_vocab(selfies_list, save_dir):
    # Generates the selfies vocab given a list of selfie strings
    # Note that the wildcard token is not a part of the selfies vocabulary
    vocab_selfies = sf.get_alphabet_from_selfies(selfies_list)
    vocab_selfies.add('[nop]') # special ignore/pad token
    vocab_selfies.add('[mask]') # special mask token

    selfies_stoi = {s:i for i, s in enumerate(vocab_selfies)}

    with open('%s/vocab_selfies.pl' % save_dir, 'wb') as vocab_file:
        pl.dump(selfies_stoi, vocab_file)

def main():
    base_smiles = '[3*]c1ccc(CN2CCC3(CC2)OCCO3)cc1'
    pdb.set_trace()
    parse_smiles_attachment(smiles=base_smiles)

# def main():
#     with open('mol_explore/data/chembl/fragment_subset.json', 'r+') as fragment_file:
#         fragment_list = json.load(fragment_file)
#
#     n_skipped = 0
#     skipped_2 = 0
#     with open('mol_explore/data/chembl/fragment_subset_gold.pkl', 'wb') as output_file:
#         base_smiles_clean_list, attach_idx_list, fragment_smiles_list = [], [], []
#         for base_smiles, fragment_smiles in tqdm(fragment_list):
#
#             output = parse_smiles_attachment(base_smiles)
#             if output is None:
#                 n_skipped += 1
#                 continue
#             base_smiles_clean, attach_idx = output
#             base_smiles_clean = clean_mapped_smiles(base_smiles_clean)
#
#             try:
#                 check_smiles = add_attachment(base_smiles_clean, attach_idx)
#             except:
#                 skipped_2 += 1
#                 continue
#
#             if check_smiles != clean_base(base_smiles):
#                 skipped_2 += 1
#                 continue
#
#             base_smiles_clean_list.append(base_smiles_clean)
#             attach_idx_list.append(attach_idx)
#             fragment_smiles_list.append(fragment_smiles)
#
#         n1 = np.array(base_smiles_clean_list)
#         n2 = np.array(attach_idx_list)
#         n3 = np.array(fragment_smiles_list)
#
#         print('Total skipped: %d' % (n_skipped))
#         print('Total unmatched: %d' % (skipped_2))
#         pdb.set_trace()
#         pl.dump([n1, n2, n3], output_file)


# def main():
#     input_path = 'data/chembl/all.txt'
#     smiles_list = []
#     with open(input_path, 'r+') as input_file:
#         for line in input_file.readlines():
#             smiles_list.append(line.strip())
#
#     fragment_list = p_umap(fragment_molecule, smiles_list)
#     n_output = len(fragment_list)
#
#     fragment_list = [x for x in fragment_list if x is not None]
#     n_output_filtered = len(fragment_list)
#
#     n_none = n_output_filtered - n_output
#
#     print('Number of examples with no fragment obtained: %d' % n_none)
#
#     with open('data/chembl/fragment.json', 'w') as output_file:
#         json.dump(fragment_list, output_file)
#     pdb.set_trace()


if __name__ == '__main__':
    main()
