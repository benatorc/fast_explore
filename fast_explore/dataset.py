import rdkit.Chem as Chem
import selfies as sf
from torch.utils.data import Dataset

from mol_explore.models.mol_graph import MolGraph

from mol_explore.utils.fragment_utils import clean_fragment_smiles

class SelfiesFragmentDataset(Dataset):
    def __init__(self, data, avocab, selfies_stoi, batch_size):
        self.batches = [data[i:(i+batch_size)] for i in range(0, len(data), batch_size)]
        self.avocab = avocab
        self.selfies_stoi = selfies_stoi

    def __len__(self):
        return len(self.batches)

    def encode_frags(self, frag_smiles):
        frag_smiles_clean = [clean_fragment_smiles(x) for x in frag_smiles]
        frag_selfies = [sf.encoder(x) for x in frag_smiles_clean]

        frag_smiles_valid, frag_selfies_valid = [], []
        # Filter out examples that resulted in None in frag_selfies:
        for i in range(len(frag_selfies)):
            if frag_selfies[i] is not None:
                frag_smiles_valid.append(frag_smiles[i])
                frag_selfies_valid.append(frag_selfies[i])

        frag_smiles, frag_selfies = frag_smiles_valid, frag_selfies_valid

        pad_len = max(sf.len_selfies(x) for x in frag_selfies) + 1

        selfies_enc = [
            sf.selfies_to_encoding(
                x, vocab_stoi=self.selfies_stoi, pad_to_len=pad_len, enc_type='label'
            ) for x in frag_selfies]

        selfies_mask = []
        pad_token = self.selfies_stoi['[nop]']
        for i, selfies in enumerate(frag_selfies):
            cur_enc = selfies_enc[i]
            mask = [0 if x == pad_token else 1 for x in cur_enc]
            mask[mask.index(0)] = 1  # first pad is stop token
            selfies_mask.append(mask)

        graph_tensors = MolGraph.tensorize(frag_smiles, self.avocab, graph_only=True)

        return graph_tensors, frag_smiles, frag_selfies, selfies_enc, selfies_mask

    def __getitem__(self, idx):
        return self.encode_frags(self.batches[idx])
