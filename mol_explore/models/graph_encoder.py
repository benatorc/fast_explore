import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from .nnutils import *
from .mol_graph import MolGraph
from .rnn import GRU, LSTM

import pdb

class MPNEncoder(nn.Module):

    def __init__(self, rnn_type, input_size, node_fdim, hidden_size, depth):
        super(MPNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth
        self.W_o = nn.Sequential(
                nn.Linear(node_fdim + hidden_size, hidden_size),
                nn.ReLU(),
        )

        if rnn_type == 'GRU':
            self.rnn = GRU(input_size, hidden_size, depth)
        elif rnn_type == 'LSTM':
            self.rnn = LSTM(input_size, hidden_size, depth)
        else:
            raise ValueError('unsupported rnn cell type ' + rnn_type)

    def forward(self, fnode, fmess, agraph, bgraph, mask):
        h = self.rnn(fmess, bgraph)
        h = self.rnn.get_hidden_state(h)
        nei_message = index_select_ND(h, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        node_hiddens = torch.cat([fnode, nei_message], dim=1)
        node_hiddens = self.W_o(node_hiddens)

        if mask is None:
            mask = torch.ones(node_hiddens.size(0), 1, device=fnode.device)
            mask[0, 0] = 0 #first node is padding

        return node_hiddens * mask, h


class GraphEncoder(nn.Module):
    def __init__(self, avocab, rnn_type, hidden_size, depth, ignore_atom_pos=False):
        super(GraphEncoder, self).__init__()
        self.avocab = avocab
        self.hidden_size = hidden_size
        self.ignore_atom_pos = ignore_atom_pos
        if ignore_atom_pos:
            self.atom_size = atom_size = avocab.size()
        else:
            self.atom_size = atom_size = avocab.size() + MolGraph.MAX_POS
        self.bond_size = bond_size = len(MolGraph.BOND_LIST)

        self.E_a = torch.eye( avocab.size()).cuda()
        self.E_b = torch.eye( len(MolGraph.BOND_LIST)).cuda()

        if ignore_atom_pos:
            self.E_pos = None
        else:
            self.E_pos = torch.eye( MolGraph.MAX_POS).cuda()

        self.encoder = MPNEncoder(rnn_type, atom_size + bond_size, atom_size, hidden_size, depth)

    def embed_graph(self, graph_tensors):
        fnode, fmess, agraph, bgraph, _ = graph_tensors
        fnode1 = self.E_a.index_select(index=fnode[:, 0], dim=0)

        if self.ignore_atom_pos:
            hnode = fnode1
        else:
            fnode2 = self.E_pos.index_select(index=fnode[:, 1], dim=0)
            hnode = torch.cat([fnode1, fnode2], dim=-1)

        fmess1 = hnode.index_select(index=fmess[:, 0], dim=0)
        fmess2 = self.E_b.index_select(index=fmess[:, 2], dim=0)
        hmess = torch.cat([fmess1, fmess2], dim=-1)

        return hnode, hmess, agraph, bgraph

    def forward(self, graph_tensors, output_edges=False):
        tensors = self.embed_graph(graph_tensors)
        hatom, h = self.encoder(*tensors, mask=None)
        if output_edges:
            return hatom, h
        else:
            return hatom

    def encode(self, graph_tensors, edge_scope=None):
        if edge_scope is not None:
            graph_vecs, graph_edge_vecs = self(graph_tensors, True)
            graph_vecs = stack_pad_tensor([graph_vecs[st: st + le] for st, le in graph_tensors[-1]])
            graph_edge_vecs = stack_pad_tensor([graph_edge_vecs[st: st + le] for st, le in edge_scope])
            return graph_vecs, graph_edge_vecs
        else:
            graph_vecs = self(graph_tensors)
            graph_vecs = stack_pad_tensor([graph_vecs[st: st + le] for st, le in graph_tensors[-1]])
            return graph_vecs
