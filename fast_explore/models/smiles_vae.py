import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from mol_explore.models.mol_graph import MolGraph
from mol_explore.models.graph_encoder import GraphEncoder
from mol_explore.models.smiles_decoder import SelfiesDecoder
from mol_explore.models.nnutils import stack_pad_tensor

import selfies as sf

import pdb

def make_cuda(graph_tensors):
    graph_tensors = [x.cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return graph_tensors


class SelfiesVGNN(nn.Module):

    def __init__(self, args):
        super(SelfiesVGNN, self).__init__()
        self.latent_size = args.latent_size
        self.selfies_stoi = args.selfies_stoi
        self.atom_vocab = args.atom_vocab
        self.constant_var = args.constant_var
        self.z_norm = args.z_norm

        self.encoder = GraphEncoder(args.atom_vocab, args.rnn_type, args.hidden_size, args.depth,
                                    ignore_atom_pos=True)
        self.decoder = SelfiesDecoder(args.selfies_stoi, args.hidden_size, args.latent_size, args.depth,
                                      args.autoregressive,)

        self.G_mean = nn.Linear(args.hidden_size, args.latent_size)
        # Use constant variance
        if self.constant_var:
            self.G_var = nn.Linear(args.hidden_size, 1)
        else:
            self.G_var = nn.Linear(args.hidden_size, args.latent_size)
        self.W_graph = nn.Sequential( nn.Linear(args.hidden_size, args.hidden_size), nn.ReLU() )

    def encode(self, x_input, edge_scope=None):
        if edge_scope is None:
            x_z = self.encoder(x_input)
            x_z = stack_pad_tensor([x_z[st:st+le] for st,le in x_input[-1]])
            return x_z
        else:
            # x_a_z: atom embeddings
            # x_e_z: edge embeddings
            x_a_z, x_e_z = self.encoder(x_input, output_edges=True)
            x_a_z = stack_pad_tensor([x_a_z[st:st+le] for st,le in x_input[-1]])

            x_e_z = stack_pad_tensor(x_e_z[st:st+le] for st,le in edge_scope)
            return x_a_z, x_e_z

    def decode(self, x_input, mean_only=False, max_decode=20):
        x_input = make_cuda(x_input)
        x_enc = self.encode(x_input)
        x_mask = self.get_mask(x_enc)
        x_agg_enc = self.aggregate(x_enc, x_mask, agg='sum')

        x_z, _ = self.rsample(x_agg_enc, self.G_mean, self.G_var, mean_only=mean_only)
        return self.decoder.decode(x_z, max_decode=max_decode)

    def rsample(self, x, W_mean, W_var, mean_only=False):
        batch_size = x.size(0)
        z_mean = W_mean(x)
        if mean_only: return z_mean, z_mean.new_tensor([0.])

        if self.constant_var:
            z_var = W_var(x).repeat(1, self.latent_size)
        else:
            z_var = W_var(x)

        z_log_var = -torch.abs(z_var)
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean).cuda()
        x_z = z_mean + torch.exp(z_log_var / 2) * epsilon

        if self.z_norm > 0.:
            x_z = torch.clamp(x_z, min=-1 * self.z_norm, max=self.z_norm)
        return x_z, kl_loss

    def aggregate(self, x, mask, agg):
        if agg == 'mean':
            x_agg = torch.sum(x, dim=1) / torch.sum(mask, dim=1)
        elif agg == 'sum':
            x_agg = torch.sum(x, dim=1)
        else:
            print('Aggregation type for graph vecs: %s not recognized' % agg)
            assert False
        return x_agg

    def get_mask(self, x):
        mask = (torch.sum(x.detach(), dim=-1) > 0.).float().unsqueeze(2)
        return mask

    def forward(self, x_input, targets, targets_mask, beta=0., mean_only=False):
        # convert to cuda
        x_input = make_cuda(x_input)
        targets = torch.tensor(targets).cuda().long()
        targets_mask = torch.tensor(targets_mask).cuda().float()

        # encode and aggregate
        x_enc = self.encode(x_input)
        x_mask = self.get_mask(x_enc)
        x_agg_enc = self.aggregate(x_enc, x_mask, agg='sum')

        # sample from distribution
        x_z, kl_div = self.rsample(x_agg_enc, self.G_mean, self.G_var, mean_only=mean_only)

        loss, selfies_list = self.decoder(
            z_graph_vecs=x_z,
            targets=targets,
            targets_mask=targets_mask,
            decode_selfies=True)
        total_loss = loss + beta * kl_div
        stats = {
            'total_loss': total_loss.item(),
            'recon_loss': loss.item(),
            'kl_loss': kl_div.item()}
        return loss + beta * kl_div, stats, selfies_list

    def get_z(self, x_input):
        """
        get the mean embedding of the given fragment graphs.
        """
        x_enc = self.encode(x_input)
        x_mask = self.get_mask(x_enc)
        x_agg_enc = self.aggregate(x_enc, x_mask, agg='sum')

        z_mean = self.G_mean(x_agg_enc)
        if self.constant_var:
            z_var = self.G_var(x_agg_enc).repeat(1, self.latent_size)
        else:
            z_var = self.G_var(x_agg_enc)
        return z_mean, z_var

    def sample_latent(self, n_samples, max_decode=20):
        x_z = torch.randn(n_samples, self.latent_size).cuda()

        selfies_list = self.decoder.decode(x_z, max_decode=max_decode)
        smiles_list = [sf.decoder(selfies) for selfies in selfies_list]

        return smiles_list, x_z
