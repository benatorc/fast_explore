import argparse
import json
from tqdm import tqdm
from p_tqdm import p_umap

import rdkit.Chem as Chem

import mol_explore.scorers as scorers

from mol_explore.utils.fragment_utils import (
    clean_fragment_smiles, fragment_molecule, parse_smiles_attachment,
    clean_fragment_smiles, add_attachment)

import pdb

def main():
    # Given a path to a list of SMILES strings
    # Break the molecules into fragment and base pairs (potentially recursively)
    # Convert the base into base_smiles + attachment_idx
    # Check that the base can be reconstructed from base_smiles and attachment_idx
    # And that the entire molecule can be reconstructed from base_smiles + fragment
    # Create SELFIES vocab from the base + fragment selfies
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the list of input smiles')
    parser.add_argument('--skip_header', action='store_true', default=False)
    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--frag_depth', type=int, default=5,
                        help='Number of times to recursively fragment molecule')
    parser.add_argument('--frag_size', type=int, default=10,
                        help='Maximum size of the fragment')
    parser.add_argument('--frag_iters', type=int, default=5)
    parser.add_argument('--min_base_size', type=int, default=10)

    # For debugging purposes, runs the generation code in serial order
    parser.add_argument('--debug_mode', action='store_true', default=False)
    args = parser.parse_args()

    # only have one criteria for selecting bases right now

    smiles_list = []
    with open(args.data_path, 'r+') as data_file:
        for line in data_file.readlines():
            if args.skip_header:
                args.skip_header = False
                continue
            # Assumes for all
            smiles = line.strip().split(',')[0]
            smiles_list.append(smiles)

    def get_frag_data(smiles,
                      size=args.frag_size,
                      depth=args.frag_depth,
                      iters=args.frag_iters,
                      min_base_size=args.min_base_size):
        input_smiles = smiles

        data = []
        for _ in range(iters):
            smiles = input_smiles
            mol = Chem.MolFromSmiles(smiles)
            for i in range(depth):
                frag_output = fragment_molecule(mol=mol, max_fragment_size=size)
                if frag_output is None:
                    break
                base_smiles, frag_smiles = frag_output
                attachment_output = parse_smiles_attachment(smiles=base_smiles)
                if attachment_output is None or attachment_output == '':
                    break
                base_smiles_na, attach_idx = attachment_output

                valid_smiles_check = add_attachment(base_smiles_na, attach_idx)
                if valid_smiles_check is None or valid_smiles_check != base_smiles:
                    break

                frag_mol = Chem.MolFromSmiles(frag_smiles)
                if frag_mol.GetNumAtoms() - 1 <= size:
                    data.append([smiles, base_smiles, frag_smiles, base_smiles_na, attach_idx])
                if i < depth - 1:
                    # recursively define:
                    smiles = base_smiles_na
                    mol = Chem.MolFromSmiles(smiles)

        data_filtered = []  # Filter the set, so that the same base is not repeated
        base_set = set()
        for x in data:
            base_smiles = x[1]
            if base_smiles in base_set:
                continue
            base_set.add(base_smiles)
            if min_base_size != -1:
                base_mol = Chem.MolFromSmiles(base_smiles)
                if base_mol.GetNumAtoms() < min_base_size:
                    continue
            data_filtered.append(x)
        return data_filtered

    if args.debug_mode:
        frag_data = []
        for smiles in tqdm(smiles_list):
            data = get_frag_data(smiles)
            frag_data += data
    else:
        frag_data = p_umap(get_frag_data, smiles_list)

    frag_data_clean = []
    for data in frag_data:
        if data != []:
            frag_data_clean += data

    with open(args.output_path, 'w+') as output_file:
        json.dump(frag_data_clean, output_file, indent=4)

    print('len: %d' % len(frag_data_clean))

if __name__ == '__main__':
    main()
