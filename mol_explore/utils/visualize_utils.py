from __future__ import print_function
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
from rdkit.Chem import rdDepictor, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
import rdkit.Geometry.rdGeometry as Geo
import numpy as np
import math
import cairosvg

# visualization
from cairosvg import svg2png
from io import BytesIO
from PIL import Image

DEF_LEN = 500

def add_indices_to_mol(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(mol.GetNumAtoms()):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol

def prepare_mol(smiles, add_Hs=False, add_indices=False, remove_mapping=False):
    m = Chem.MolFromSmiles(smiles)
    if add_Hs:
        m = Chem.AddHs(m)
        
    if add_indices:
        mol = add_indices_to_mol(m)
        
    if remove_mapping:
        for atom in m.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        
    mc = Chem.Mol(m.ToBinary())
    try:
        Chem.Kekulize(mc)
    except:
        mc = Chem.Mol(mol.ToBinary())
    
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    return mc

def normalize_scores(scores):
    scores = (scores - np.mean(scores)) / np.std(scores)
    return scores

def draw_attention(smiles, attention_scores=[]):
    mc = prepare_mol(smiles)
    
    x, y = DEF_LEN, DEF_LEN
    drawer = rdMolDraw2D.MolDraw2DSVG(x, y, x, y)
    drawer.SetFontSize(.45)
    
    n_atoms = mc.GetNumAtoms()
    assert len(attention_scores) == n_atoms
    
    scores = np.array(attention_scores)
    scores = normalize_scores(scores)
    print(scores)

    min_score, max_score = np.min(scores), np.max(scores)
    
    norm_scores = (scores - min_score) / (max_score - min_score)
    norm_scores = (norm_scores * 0.6) + 0.20
    norm_scores = 1 - norm_scores  # Because lower scores are more intense colors
    print(norm_scores)
        
    h_atoms = []
    h_colors = {}
    for idx, score in enumerate(norm_scores):
        h_atoms.append(idx)
        h_colors[idx] = (score, 0.35 * score, 0.25 * score)
    print(h_atoms)
    print(h_colors)
    
    drawer.DrawMolecules([mc], highlightAtoms=[h_atoms], highlightAtomColors=[h_colors])
    
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # display(SVG(svg.replace('svg:', '')))

def reorder_mol(mol):
    atom_map = {}
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom_map[int(atom.GetProp('molAtomMapNumber'))] = atom.GetIdx()
    new_ordering = []
    for idx in range(len(atom_map)):
        new_ordering.append(atom_map[idx])
    if len(new_ordering) == mol.GetNumAtoms():
        return Chem.rdmolops.RenumberAtoms(mol, new_ordering)
    else:
        return mol
    
def draw_mol(smiles_list, names_list=[], highlights=[], atom_colors={}, highlight_bonds=[], bond_colors={}, logp=False, add_Hs=False, add_indices=False, remove_mapping=False, change_to_mapped_indices=False):
    smiles_list = [smiles.replace(' ', '') for smiles in smiles_list]
    mc_list = [prepare_mol(smiles, add_Hs, add_indices, remove_mapping) for smiles in smiles_list]

    if change_to_mapped_indices:
        mc_list = [reorder_mol(x) for x in mc_list]
    
    n_smiles = len(smiles_list)
    y = DEF_LEN
    x_t = DEF_LEN * n_smiles 
    x_i = DEF_LEN
    
    drawer = rdMolDraw2D.MolDraw2DSVG(x_t, y, x_i, y)    
    drawer.SetFontSize(0.5)

    drawer.DrawMolecules(mc_list, highlightAtoms=highlights, highlightAtomColors=atom_colors, highlightBonds=highlight_bonds, highlightBondColors=bond_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    png = svg2png(bytestring=svg)
    pil_img = Image.open(BytesIO(png)).convert('RGB')
    # display(SVG(svg.replace('svg:','')))
    return np.array(pil_img)

def draw_reaction(rxn_smiles='', highlights=[], remove_mapping=False):
    reactants = rxn_smiles.split('>>')[0]
    products = rxn_smiles.split('>>')[1]
    
    if highlights == []:
        draw_mol([reactants, products], ['Reactants', 'Products'], remove_mapping=remove_mapping)
    else:
        draw_mol([reactants, products], ['Reactants', 'Products'], highlights=highlights, remove_mapping=remove_mapping)
        