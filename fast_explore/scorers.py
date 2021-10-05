import pickle as pl
import numpy as np

import rdkit.Chem.QED as QED
from rdkit.Chem.Descriptors import MolLogP
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

import networkx as nx

from mol_explore.utils import sascorer

from p_tqdm import p_umap

class diversity_score():
    def __init__(self):
        return

    def __call__(self, s1, s2):
        m1 = [Chem.MolFromSmiles(x) for x in s1]
        m2 = [Chem.MolFromSmiles(x) for x in s2]

        fp1 = [diversity_score.fingerprints_from_mol(m) for m in m1]
        fp2 = [diversity_score.fingerprints_from_mol(m) for m in m2]

        fp1 = np.concatenate(fp1)
        fp2 = np.concatenate(fp2)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)


class constrained_logP_model():
    def __init__(self, normalized_score=True):
        self.normalized_score = normalized_score
    
    def penalized_logp(self, mol):
        """
        Reward that consists of log p penalized by SA and # long cycles,
        as described in (Kusner et al. 2017). Scores are normalized based on the
        statistics of 250k_rndm_zinc_drugs_clean.smi dataset
        :param mol: rdkit mol object
        :return: float
        """
        # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
        logP_mean = 2.4570953396190123
        logP_std = 1.434324401111988
        SA_mean = -3.0525811293166134
        SA_std = 0.8335207024513095
        cycle_mean = -0.0485696876403053
        cycle_std = 0.2860212110245455

        log_p = MolLogP(mol)
        SA = -sascorer.calculateScore(mol)

        # cycle score
        cycle_list = nx.cycle_basis(nx.Graph(
            Chem.rdmolops.GetAdjacencyMatrix(mol)))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        cycle_score = -cycle_length

        if self.normalized_score:
            normalized_log_p = (log_p - logP_mean) / logP_std
            normalized_SA = (SA - SA_mean) / SA_std
            normalized_cycle = (cycle_score - cycle_mean) / cycle_std

            return normalized_log_p + normalized_SA + normalized_cycle
        else:
            return log_p + SA + cycle_score

    def __call__(self, smiles_list):
        mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
        scores = [self.penalized_logp(x) for x in mol_list]
        return np.array(scores)

class gsk3_rf_model():
    """Scores based on an ECFP classifier for activity."""

    def __init__(self, pl_path='/data/rsg/chemistry/bensonc/mol_explore/mol_explore/data/gsk3/gsk3.pkl'):
        with open(pl_path, "rb") as f:
            self.clf = pl.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append( int(mol is not None) )
            fp = gsk3_rf_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return {'gsk': scores, }

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)

class gsk3_qed_rf_model():
    def __init__(self):
        self.gsk_model = gsk3_rf_model()
        self.qed_scorer = qed_func()
    
    def __call__(self, smiles_list):
        gsk_scores = self.gsk_model(smiles_list)
        qed_scores = self.qed_scorer(smiles_list)

        return {**gsk_scores, **qed_scores}

class gsk3_qed_sa_rf_model():
    def __init__(self):
        self.gsk_model = gsk3_rf_model()
        self.sa_scorer = sa_func()
        self.qed_scorer = qed_func()
    
    def __call__(self, smiles_list):
        gsk_scores = self.gsk_model(smiles_list)
        sa_scores = self.sa_scorer(smiles_list)
        qed_scores = self.qed_scorer(smiles_list)

        return {**gsk_scores, **sa_scores, **qed_scores }

class jnk3_rf_model():
    """Scores based on an ECFP classifier for activity."""

    def __init__(self, pl_path='/data/rsg/chemistry/bensonc/mol_explore/mol_explore/data/jnk3/jnk3.pkl'):
        with open(pl_path, "rb") as f:
            self.clf = pl.load(f)

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i,smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append( int(mol is not None) )
            fp = jnk3_rf_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return {'jnk': scores, }

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)

class jnk3_qed_sa_rf_model():
    def __init__(self):
        self.jnk_model = jnk3_rf_model()
        self.sa_scorer = sa_func()
        self.qed_scorer = qed_func()
    
    def __call__(self, smiles_list):
        jnk_scores = self.jnk_model(smiles_list)
        sa_scores = self.sa_scorer(smiles_list)
        qed_scores = self.qed_scorer(smiles_list)

        return {**jnk_scores, **sa_scores, **qed_scores, }

class gsk3_jnk3_model():
    def __init__(self):
        self.gsk_model = gsk3_rf_model()
        self.jnk_model = jnk3_rf_model()

    def __call__(self, smiles_list):
        gsk_scores = self.gsk_model(smiles_list)
        jnk_scores = self.jnk_model(smiles_list)

        return {**gsk_scores, **jnk_scores, }

class gsk3_jnk3_qed_sa_model():
    def __init__(self):
        self.gsk_model = gsk3_rf_model()
        self.jnk_model = jnk3_rf_model()
        self.sa_scorer = sa_func()
        self.qed_scorer = qed_func()

    def __call__(self, smiles_list):
        gsk_scores = self.gsk_model(smiles_list)
        jnk_scores = self.jnk_model(smiles_list)
        sa_scores = self.sa_scorer(smiles_list)
        qed_scores = self.qed_scorer(smiles_list)

        return {**gsk_scores, **jnk_scores, **sa_scores, **qed_scores, }


class qed_func():
    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(0)
            else:
                scores.append(QED.qed(mol))
        return {'qed': np.float32(scores)}

class sa_func():
    # SA score is normally from 1 to 10, with 1 being the best
    # rescale so it is in [0, 1], with higher being better
    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(10)
            else:
                scores.append(sascorer.calculateScore(mol))
        scores = np.float32(scores)
        scores = 1 - (scores / 10.)
        return {'sa': scores}