import numpy as np
import torch
import torch.nn as nn
import mol_explore.rl.utils as utils


PAD_LOGIT = -1e10
class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class TypeHead(nn.Module):
    
    def __init__(self, node_fdim, hidden_dim, hidden_depth):
        super().__init__()
        # two action types: delete / add
        self.trunk = utils.mlp(node_fdim, hidden_dim, 2, hidden_depth)
    
    def forward(self, obs, no_avail_delete, no_avail_attach, action_max, action_min):
        """
        if no deletion action available, give prob=1 for addition and vice versa.
        """
        logits = self.trunk(obs)
        logits[no_avail_attach, 0] = PAD_LOGIT
        logits[no_avail_delete, 1] = PAD_LOGIT
        if action_max < 1.0 or action_min > 0.0:
            probs = torch.softmax(logits, dim=1)
            probs = torch.clip(probs, min=action_min, max=action_max)
            dist = utils.FixedCategorical(probs=probs)
        else:
            dist = utils.FixedCategorical(logits=logits)
        return dist

class SelectionHead(nn.Module):
    """
    TODO: can also consider using a Pointer Network here
    """
    def __init__(self, node_fdim, hidden_dim, hidden_depth):
        super().__init__()
        self.trunk = utils.mlp(node_fdim, hidden_dim, 1, hidden_depth)

    def forward(self, obs, masks):
        logits = self.trunk(obs.view(-1, obs.shape[-1]))
        logits = logits.view(obs.shape[:-1])
        logits.masked_fill_(~masks, PAD_LOGIT)
        no_available = masks.sum(-1) == 0
        logits[no_available, 0] = 1e10  # if no attachment point avaiable, choose the first
        dist = utils.FixedCategorical(logits=logits)
        return dist, no_available
    
class ContinuousFragHead(nn.Module):
    """
    TODO: can either use the attachment point node embedding only or use an aggregated graph embedding
    """
    def __init__(self, action_dim, node_fdim, hidden_dim, hidden_depth, log_std_bounds, log_std_init=-1.0):
        super().__init__()
        # TODO Instead, use the fixed variance for all actions in PPO-A2C...
        self.trunk = utils.mlp(node_fdim, hidden_dim, action_dim, hidden_depth)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init, requires_grad=True)
        self.log_std_bounds = log_std_bounds

    def forward(self, obs):
        # mu, log_std = self.trunk(obs)
        log_std = torch.tanh(self.log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()
        mu = self.trunk(obs)
        # std = self.log_std.exp()
        # dist = utils.SquashedNormal(mu, std)  # Tanh transform to [-1,1]
        dist = torch.distributions.Normal(mu, std)
        return dist

class DiscreteFragHead(nn.Module):
    def __init__(self, action_dim, n_embed, node_fdim, hidden_dim, hidden_depth):
        super().__init__()
        self.action_dim = action_dim
        self.n_embed = n_embed
        self.trunk = utils.mlp(node_fdim, hidden_dim, action_dim * n_embed, hidden_depth)
        
    def forward(self, obs):
        return utils.FixedCategorical(logits=self.trunk(obs).view(-1, self.action_dim, self.n_embed))

class FixedFragHead(nn.Module):
    def __init__(self, action_dim, node_fdim, hidden_dim, hidden_depth):
        super().__init__()
        self.action_dim = action_dim

        self.trunk = utils.mlp(node_fdim, hidden_dim, action_dim, hidden_depth)

    def forward(self, obs):
        return utils.FixedCategorical(logits=self.trunk(obs).view(-1, self.action_dim))

class Actor(nn.Module):
    def __init__(self, encoder, action_dim, n_embed, node_fdim, hidden_dim, hidden_depth, frag_input, action_type, action_max, action_min):
        super().__init__()
        self.encoder = DataParallelPassthrough(encoder)

        assert frag_input in ['node', 'graph', 'node+graph']
        self.frag_input = frag_input
        self.action_dim = action_dim
        self.n_embed = n_embed

        self.action_max = action_max
        self.action_min = action_min

        self.type_head = TypeHead(node_fdim, hidden_dim, hidden_depth)
        self.deletion_head = SelectionHead(node_fdim, hidden_dim, hidden_depth)
        self.attachment_head = SelectionHead(node_fdim, hidden_dim, hidden_depth)
        
        self.action_type = action_type
        if action_type == 'discrete':
            if frag_input in ['node', 'graph']:
                self.fragment_head = DiscreteFragHead(action_dim, n_embed, node_fdim, hidden_dim, hidden_depth)
            else:
                self.fragment_head = DiscreteFragHead(action_dim, n_embed, node_fdim * 2, hidden_dim, hidden_depth)
        elif action_type == 'fixed':
            self.fragment_head = FixedFragHead(action_dim, node_fdim * 2, hidden_dim, hidden_depth)
        else:
            print('Unimplemented action type for actor: %s' % action_type)
            exit()

    def forward(self, obs, del_mask, add_mask, edge_scope=None, attachment=None, detach=False):
        # graph pooling: sum vs mean
        if detach:
            obs = obs.detach()
               
        node_obs, edge_obs = self.encoder.encode(obs, edge_scope)  # batchsize x <mol_graph>
        deletion_dists, no_avail_delete = self.attachment_head(edge_obs, del_mask)
        attachment_dists, no_avail_attach = self.attachment_head(node_obs, add_mask)
        type_dists = self.type_head(node_obs.mean(dim=1), no_avail_delete, no_avail_attach, action_max=self.action_max, action_min=self.action_min)
        
        if attachment is None:
            attachment = attachment_dists.sample().flatten()
        if self.action_type == 'discrete':
            if self.frag_input == 'node':
                attach_nodes = obs[np.arange(node_obs.shape[0]), attachment]
                fragment_dists = self.fragment_head(attach_nodes)
            elif self.frag_input == 'graph':
                fragment_dists = self.fragment_head(node_obs.mean(dim=1))
            else:
                attach_nodes = node_obs[np.arange(node_obs.shape[0]), attachment]
                frag_input = torch.cat([node_obs.mean(dim=1), attach_nodes], dim=1)
                fragment_dists = self.fragment_head(frag_input)
        elif self.action_type == 'fixed':
            attach_nodes = node_obs[np.arange(node_obs.shape[0]), attachment]
            frag_input = torch.cat([node_obs.mean(dim=1), attach_nodes], dim=1)
            fragment_dists = self.fragment_head(frag_input)

        return type_dists, deletion_dists, attachment_dists, fragment_dists

class Critic(nn.Module):
    def __init__(self, encoder, node_fdim, hidden_dim, hidden_depth):
        super().__init__()
        self.encoder = DataParallelPassthrough(encoder)
        self.V = utils.mlp(node_fdim, hidden_dim, 1, hidden_depth)

    def forward(self, obs, detach=False):
        if detach:
            obs = obs.detach()
        obs = self.encoder.encode(obs)  # batchsize x <mol_graph>
        # TODO mean pooling is questionable
        return self.V(obs.mean(dim=1))
