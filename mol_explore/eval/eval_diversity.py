import json
import argparse
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import AllChem
import random

from tqdm import tqdm

from p_tqdm import p_umap

from mol_explore.scorers import (
        gsk3_rf_model, jnk3_rf_model, gsk3_qed_sa_rf_model,
        jnk3_qed_sa_rf_model, gsk3_jnk3_model, gsk3_jnk3_qed_sa_model)
from mol_explore.rl.rewards import prop_qualified, TASK2SF

import pdb

ACTIVES = {
    'gsk': 'data/gsk3/actives.txt',
    'jnk': 'data/jnk3/actives.txt',
    'gsk+jnk': 'data/gsk3_jnk3/actives.txt',
}

def clean_smiles(smiles):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        return ''

def eval_metrics(smiles_list, actives_list, score_func, fp_type):
    scores_dict = score_func(smiles_list)

    success = prop_qualified(scores_dict)
    success_rate = np.mean(success.astype(float))

    smiles_actives = []
    for i in range(len(smiles_list)):
        if success[i]:
            smiles_actives.append(smiles_list[i])

    fps_list = get_fps(smiles_actives, fp_type=fp_type)
    actives_fps = get_fps(actives_list, fp_type=fp_type)
    novelty = compute_novelty(actives_fps, fps_list)

    diversity = compute_diversity(fps_list) 
    return success_rate, novelty, diversity

# use L2 to approximate
def get_fps(smiles_list, fp_type='morgan', fp_rad=3, fp_dim=2048, use_chirality=False):
    mols = [Chem.MolFromSmiles(x) for x in smiles_list]
    clean_mols = []
    for mol in mols:
        if mol is None:
            clean_mols.append(Chem.MolFromSmiles(''))
        else:
            clean_mols.append(mol)
    assert len(clean_mols) == len(mols)
    mols = clean_mols
    if fp_type == 'morgan':
        fps = [AllChem.GetMorganFingerprintAsBitVect(
            x, radius=fp_rad, nBits=fp_dim, useChirality=use_chirality) for x in mols]
    elif fp_type == 'maccs':
        fps = [MACCSkeys.GenMACCSKeys(x) for x in mols]
    elif fp_type == 'ap':
        fps = [Pairs.GetAtomPairFingerprint(x) for x in mols]
    else:
        print('Fingerprint type: %s not recognized' % fp_type)
        assert False
    return fps

def fp2numpy(fp):
    return np.unpackbits(np.frombuffer(DataStructs.BitVectToBinaryText(fp), dtype=np.uint8), bitorder='little')


def get_sims_from_fps(source_fps, target_fps_list):
    return DataStructs.BulkTanimotoSimilarity(source_fps, target_fps_list)

def compute_diversity(fps_list):
    n = len(fps_list)

    inputs = []
    for i in range(n):
        inputs.append((fps_list[i], fps_list[:i]))

    def bulk_sim(fps_inputs):
        fps_source, fps_targets = fps_inputs
        return DataStructs.BulkTanimotoSimilarity(fps_source, fps_targets)

    outputs = p_umap(bulk_sim, inputs)
    outputs = sum(outputs, [])
    sum_scores = sum(outputs)
    n_pairs = n * (n-1) / 2

    return 1 - sum_scores / n_pairs

def compute_novelty(source_list, target_list):
    n_sim = 0
    for fps in target_list:
        sims = DataStructs.BulkTanimotoSimilarity(fps, source_list)
        if max(sims) >= 0.4:
            n_sim += 1
    return 1 - n_sim / len(target_list)

def get_actives(task):
    if 'gsk' not in task:
        actives_path = ACTIVES['jnk']
    elif 'jnk' not in task:
        actives_path = ACTIVES['gsk']
    else:
        actives_path = ACTIVES['gsk+jnk']

    actives_list = []
    with open(actives_path, 'r') as actives_file:
        skip_header = True
        for line in actives_file.readlines():
            if skip_header:
                skip_header = False
                continue
            actives_list.append(line.split(',')[0])
    print('Retrieving actives from: %s (%d)' % (actives_path, len(actives_list)))
    return actives_list

def get_smiles(path, input_type, max_inputs=5000):
    print('Loading smiles from: %s' % path)
    if input_type == 'json':
        with open(path, 'r') as f:
            smiles_list = json.load(f)
    elif input_type == 'np':
        frontier = np.load(path, allow_pickle=True).item()
        smiles_list = frontier.discovered['smiles']
    elif input_type == 'txt':
        with open(path, 'r') as f:
            smiles_list = []
            for line in f.readlines():
                smiles_list.append(line.strip().split(' ')[0])
    else:
        assert False
    
    smiles_list = [clean_smiles(x) for x in smiles_list]
    smiles_set = set(smiles_list)
    print('Num smiles loaded: %d, unique: %d' % (len(smiles_list), len(smiles_set)))

    if len(smiles_list) > max_inputs:
        smiles_list = smiles_list[:max_inputs]
    
    return smiles_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--input_type', type=str, choices=['json', 'np', 'txt'], default='json')
    parser.add_argument('--max_inputs', type=int, default=5000)
    parser.add_argument('--fps_type', type=str, choices=['morgan', 'maccs', 'ap'], default='morgan')
    args = parser.parse_args()

    score_func = TASK2SF[args.task]()
    actives_list = get_actives(args.task)
    smiles_list = get_smiles(args.path, args.input_type, args.max_inputs)

    success, novelty, diversity = eval_metrics(smiles_list, actives_list, score_func, args.fps_type)

    print('Success: %.3f, Novelty: %.3f, Diversity: %.3f' % (
        success, novelty, diversity))

if __name__ == '__main__':
    main()