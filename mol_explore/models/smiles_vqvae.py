import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from mol_explore.models.mol_graph import MolGraph
from mol_explore.models.graph_encoder import GraphEncoder
from mol_explore.models.smiles_decoder import SelfiesDecoder
from mol_explore.models.nnutils import stack_pad_tensor

from mol_explore.models.nearest_embed import NearestEmbed

import selfies as sf

import pdb

def make_cuda(graph_tensors):
    graph_tensors = [x.cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return graph_tensors

class SelfiesVQVGNN(nn.Module):

    def __init__(self, args):
        super(SelfiesVQVGNN, self).__init__()
        self.latent_size = args.latent_size
        self.selfies_stoi = args.selfies_stoi
        self.atom_vocab = args.atom_vocab
        self.constant_var = args.constant_var
        self.hidden_size = args.hidden_size
        self.n_embed = args.n_embed
        self.vq_coef = args.vq_coef
        self.commit_coef = args.commit_coef

        self.encoder = GraphEncoder(args.atom_vocab, args.rnn_type, args.hidden_size, args.depth,
                                    ignore_atom_pos=True)
        self.decoder = SelfiesDecoder(args.selfies_stoi, args.hidden_size, args.hidden_size, args.depth,
                                      args.autoregressive,)

        self.G_mean = nn.Linear(args.hidden_size, args.latent_size)
        # Use constant variance
        if self.constant_var:
            self.G_var = nn.Linear(args.hidden_size, 1)
        else:
            self.G_var = nn.Linear(args.hidden_size, args.latent_size)
        self.W_graph = nn.Sequential( nn.Linear(args.hidden_size, args.hidden_size), nn.ReLU() )

        self.embed_size = 2 * self.n_embed
        self.W_enc_out = nn.Linear(args.hidden_size, self.embed_size * args.latent_size)
        self.emb = NearestEmbed(args.n_embed, self.embed_size)

        self.W_dec_in = nn.Linear(self.embed_size * self.latent_size, args.hidden_size)

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
        x_enc = self.W_enc_out(x_agg_enc).reshape(-1, self.n_embed, self.hidden_size)

        z_q, _ = self.emb(x_enc.permute(0, 2, 1), weight_sg=True)
        z_q_avg = z_q.permute(0, 2, 1).mean(dim=1)

        return self.decoder.decode(z_q_avg, max_decode=max_decode)

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

        # overwrite
        x_enc = self.W_enc_out(x_agg_enc).reshape(-1, self.latent_size, self.embed_size)
        z_q, _ = self.emb(x_enc.permute(0, 2, 1), weight_sg=True)
        emb, _ = self.emb(x_enc.permute(0, 2, 1).detach())

        z_q_avg = self.W_dec_in(z_q.permute(0, 2, 1).contiguous().view(-1, self.latent_size * self.embed_size))

        loss, selfies_list = self.decoder(
            z_graph_vecs=z_q_avg,
            targets=targets,
            targets_mask=targets_mask,
            decode_selfies=True,)

        vq_loss = F.mse_loss(emb, x_enc.permute(0, 2, 1).detach())
        commit_loss = F.mse_loss(x_enc.permute(0, 2, 1), emb.detach())

        total_loss = loss + self.vq_coef * vq_loss + self.commit_coef * commit_loss

        stats = {
            'total_loss': total_loss.item(),
            'recon_loss': loss.item(),
            'vq_loss': vq_loss.item(),
            'commit_loss': commit_loss.item(),}

        return total_loss, stats, selfies_list


    def get_z(self, x_input):
        """
        get the mean embedding of the given fragment graphs.
        """
        x_enc = self.encode(x_input)
        atom_mask = self.get_mask(x_enc)
        x_agg_enc = self.aggregate(x_enc, atom_mask, agg='sum')
        x_enc = self.W_enc_out(x_agg_enc).reshape(-1, self.latent_size, self.embed_size)
        z_q, discrete_code = self.emb(x_enc.permute(0, 2, 1), weight_sg=True)
        z_q_avg = self.W_dec_in(z_q.permute(0, 2, 1).contiguous().view(-1, self.latent_size * self.embed_size))

        return z_q_avg, discrete_code

    def sample_latent(self, n_samples, max_decode=20):
        x_enc = torch.randn(n_samples, self.latent_size, self.embed_size).cuda()
        emb, _ = self.emb(x_enc.permute(0, 2, 1))

        z_q_avg = self.W_dec_in(emb.permute(0, 2, 1).contiguous().view(-1, self.latent_size * self.embed_size))

        selfies_list = self.decoder.decode(z_q_avg, max_decode=max_decode)
        smiles_list = ['*' + sf.decoder(selfies) for selfies in selfies_list]
        return smiles_list, z_q_avg

    def decode_action(self, input):
        """
        decode actions from categorical latents [B x latent_size x 1 <Categorical(n_embed)>]
        """
        B = input.shape[0]
        with torch.no_grad():
            emb = self.emb.weight.t().index_select(0, input.view(-1)).view([B, self.latent_size * self.embed_size])
            z_q_avg = self.W_dec_in(emb.contiguous().view(-1, self.latent_size * self.embed_size))
            selfies_list = self.decoder.decode(z_q_avg, max_decode=20)

            smiles_list = []
            for selfies in selfies_list:
                smiles_decode = sf.decoder(selfies)
                if smiles_decode is None:
                    smiles_decode = ''
                smiles_list.append('*' + smiles_decode)
            return {'z': z_q_avg, 'smiles': smiles_list}
            # try:
            #     return {'z': z_q_avg,
            #             'smiles': ['*' + sf.decoder(selfies) for selfies in selfies_list]}
            # except:
            #     import pdb
            #     pdb.set_trace()
