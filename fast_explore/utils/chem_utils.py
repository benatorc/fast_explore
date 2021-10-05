from mol_explore.models.chemutils import bond_match
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import selfies as sf
from mol_explore.models.mol_graph import MolGraph

import pdb

def selfies_frag_to_smiles(selfies):
    try:
        smiles = sf.decoder(selfies)
    except:
        smiles = None

    if smiles is None:
        return ''
    smiles = '*' + smiles
    try:
        smiles_clean = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        return ''
    return smiles_clean


def compute_fingerprint_score(mol1, mol2):
    mol_pair = [mol1, mol2]
    fps_pair = [Chem.RDKFingerprint(m) for m in mol_pair]

    sim = DataStructs.FingerprintSimilarity(fps_pair[0], fps_pair[1])
    return sim


def add_atoms_with_attachment(emol, mol):
    # Add the atoms from mol to emol, and extracts the attachment atoms

    # Mapping from original mol to new mol
    a_map = {}

    attach_idx = -1
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()

        if atom_symbol == '*':
            neighbors = atom.GetNeighbors()
            assert len(neighbors) == 1  # wildcard atom should only have 1 neighbor
            attach_idx = neighbors[0].GetIdx()
            continue
        atom_fc = atom.GetFormalCharge()

        new_atom = Chem.Atom(atom_symbol)
        new_atom.SetFormalCharge(atom_fc)
        if atom_symbol != 'C':
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())

        new_atom_idx = emol.AddAtom(new_atom)
        a_map[atom_idx] = new_atom_idx
    assert attach_idx != -1

    return a_map, attach_idx

def add_bonds(emol, mol, a_map):
    # Add the bonds from mol to emol using the mapping from a_map

    for bond in mol.GetBonds():
        # Ignore the wildcard atoms
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        a1_idx, a2_idx = a1.GetIdx(), a2.GetIdx()
        if a1.GetSymbol() == '*' or a2.GetSymbol() == '*':
            continue
        bond_type = bond.GetBondType()
        emol.AddBond(a_map[a1_idx], a_map[a2_idx], bond_type)


def combine_smiles(s1, s2):
    # s1, s2 are smile strings with * tokens marking the attachment atom
    # The attachment bond will be default to a single bond

    m1 = Chem.MolFromSmiles(s1)
    m2 = Chem.MolFromSmiles(s2)

    emol = Chem.RWMol()

    a1_map, a1_idx = add_atoms_with_attachment(emol, m1)
    a2_map, a2_idx = add_atoms_with_attachment(emol, m2)

    add_bonds(emol, m1, a1_map)
    add_bonds(emol, m2, a2_map)

    emol.AddBond(a1_map[a1_idx], a2_map[a2_idx], Chem.rdchem.BondType.SINGLE)

    new_smiles = Chem.MolToSmiles(emol)

    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(new_smiles))
    except:
        return None

def delete_bond(smiles, e_idx):
    mol_graph = MolGraph(smiles)

    edges = mol_graph.mol_graph.edges
    for idx, (u, v) in enumerate(edges):
        if e_idx == idx:
            break

    mol = Chem.MolFromSmiles(smiles)
    emol = Chem.RWMol(mol)
    
    # try:
    emol.RemoveBond(u, v)

    mapping = []
    frags = list(Chem.rdmolops.GetMolFrags(emol,
        asMols=True, fragsMolAtomMapping=mapping))

    if len(frags) == 1:
        return Chem.MolToSmiles(emol)

    if u in mapping[0]:
        return Chem.MolToSmiles(frags[0])
    else:
        return Chem.MolToSmiles(frags[1])
    # except:
    #     print('Cannot delete bond for smiles: %s bond u (%d) -> v (%d)' % (
    #         smiles, u, v))

# def delete_bond(smiles, u, v):
#     mol = Chem.MolFromSmiles(smiles)

#     emol = Chem.RWMol(mol)
#     try:
#         emol.RemoveBond(u, v)

#         mapping = []
#         frags = list(Chem.rdmolops.GetMolFrags(emol,
#             asMols=True, fragsMolAtomMapping=mapping))

#         if len(frags) == 1:
#             return Chem.MolToSmiles(emol)

#         if u in mapping[0]:
#             return Chem.MolToSmiles(frags[0])
#         else:
#             return Chem.MolToSmiles(frags[1])
#     except:
#         print('Cannot delete bond for smiles: %s bond u (%d) -> v (%d)' % (
#             smiles, u, v))

def get_scaffold_dict(smiles_list):
    # Takes in a list of smiles, outputs a mapping of the scaffold smiles to
    # a list of the smiles

    scaffold_dict = {}
    for mol_smiles in smiles_list:
        scaffold_smiles = MurckoScaffold.MurckoScaffoldSmiles(
            smiles=mol_smiles, includeChirality=False)
        if scaffold_smiles in scaffold_dict:
            scaffold_dict[scaffold_smiles].append(mol_smiles)
        else:
            scaffold_dict[scaffold_smiles] = [mol_smiles]
    return scaffold_dict


if __name__ == '__main__':
    main()
