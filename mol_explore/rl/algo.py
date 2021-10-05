import torch
import torch.nn as nn

from mol_explore.models.vocab import common_atom_vocab
from mol_explore.models.mol_graph import MolGraph
from mol_explore.models.smiles_vae import make_cuda

class PPOAgent(object):

    def __init__(
        self,
        encoder, actor, critic,
        batch_size, ppo_epoch, num_mini_batch,
        clip_param, value_loss_coef, entropy_coef, type_entropy_coef,
        lr, eps, max_grad_norm, device, match_dist
    ):

        # Params
        self.device = device
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.type_entropy_coef = type_entropy_coef
        self.max_grad_norm = max_grad_norm
        self.match_dist = match_dist

        self.encoder = encoder.to(self.device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.ac_params = set(self.encoder.parameters()) | \
            set(self.actor.parameters()) | set(self.critic.parameters())
        self.ac_optimizer = torch.optim.Adam(
            self.ac_params, lr=lr, eps=eps)

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def save_checkpoint(self, model_dir, step):
        state = {
            'step': step,
            'encoder': self.encoder.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }
        print(f'Saving model to: {model_dir}')
        torch.save(state, model_dir)

    def load_checkpoint(self, model_dir):
        state = torch.load(f'{model_dir}')
        self.encoder.load_state_dict(state['encoder'])
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        return state['step']

    def act(self, obs, sample=True):
        dist_type, dist_del, dist_attach, dist_frag = self.actor(
                obs['base'], obs['del_mask'], obs['add_mask'], obs['edge_scope'])
        action_type = dist_type.sample() if sample else dist_type.mode()
        action_del = dist_del.sample() if sample else dist_del.mode()
        action_attach = dist_attach.sample() if sample else dist_attach.mode()
        action_frag = dist_frag.sample() if sample else dist_frag.mode()

        action = {
            'type': action_type,
            'type_probs': dist_type.probs.cpu().numpy(),
            'deletion': action_del,
            'deletion_probs': dist_del.probs.cpu().numpy(),
            'attachment': action_attach,
            'attach_probs': dist_attach.probs.cpu().numpy(),
            'fragment': action_frag
        }
        return action

    def get_value(self, obs):
        return self.critic(obs['base'])

    def ac_evaluate(self, obs, action):
        value = self.critic(obs['base'])
        dist_type, dist_del, dist_attach, dist_frag = self.actor(
            obs=obs['base'], 
            del_mask=obs['del_mask'], 
            add_mask=obs['add_mask'], 
            edge_scope=obs['edge_scope'])
        
        # prob of action type
        type_log_probs = dist_type.log_probs(action['type'])
        # type_log_probs = torch.clip(type_log_probs, torch.log(torch.tensor(.5)).cuda())
        # type_log_probs = torch.clip(type_log_probs, min=torch.log(torch.tensor(.2), max=torch.log(torch.tensor(.8)).cuda())
        del_log_probs = dist_del.log_probs(action['deletion'])
        add_log_probs = dist_attach.log_probs(action['attachment']) + \
            dist_frag.log_probs(action['fragment'])
            
        type_entropy = dist_type.entropy()
        del_entropy = dist_del.entropy()
        add_entropy = dist_attach.entropy() + dist_frag.entropy().mean(-1)  # NOTE use mean instead of sum. may want to change.
        
        add_actions = (action['type'] == 0).float()
        action_log_probs = type_log_probs + add_actions * add_log_probs + (1-add_actions) * del_log_probs
        dist_entropy = self.type_entropy_coef * type_entropy + add_actions.squeeze() * add_entropy + (1-add_actions).squeeze() * del_entropy
        dist_entropy = dist_entropy.mean()
        return value, action_log_probs, dist_entropy
            
    def update(self, rollouts, logger, step):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        total_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

            for sample in data_generator:
                (obs_batch, action_batch, value_pred_batch, return_batch,
                    mask_batch, old_action_log_prob_batch, adv_targ) = sample

                # need to acquire obs tensors
                obs_graph, masks = MolGraph.tensorize_action(obs_batch, avocab=common_atom_vocab)
                obs_graph = make_cuda(obs_graph)
                add_mask, del_mask, edge_scope = masks
                obs_tensor = {'base': obs_graph, 'add_mask': add_mask.cuda(), 'del_mask': del_mask.cuda(), 'edge_scope': edge_scope}
                
                # debug NaN errors.
                if torch.isnan(self.actor.encoder.encoder.W_o[0].weight).any():
                    import pdb
                    pdb.set_trace()

                values, action_log_probs, dist_entropy = self.ac_evaluate(obs_tensor, action_batch)
                ratio = torch.exp(action_log_probs -
                                old_action_log_prob_batch) 
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_pred_batch + \
                    (values - value_pred_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                            value_losses_clipped).mean()

                self.ac_optimizer.zero_grad()
                total_loss = (action_loss + value_loss * self.value_loss_coef -
                              dist_entropy * self.entropy_coef)
                total_loss.backward()

                if torch.isnan(self.actor.encoder.encoder.W_o[0].weight).any():
                    import pdb
                    pdb.set_trace()

                ac_grad = nn.utils.clip_grad_norm_(self.ac_params, self.max_grad_norm)
                self.ac_optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                total_loss_epoch += total_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        total_loss_epoch /= num_updates

        logger.scalar('train/critic_loss', value_loss_epoch)
        logger.scalar('train/actor_loss', action_loss_epoch)
        logger.scalar('train/total_loss', total_loss_epoch)
        logger.scalar('train/ac_gradient', ac_grad)
        logger.scalar('train/actor_entropy', dist_entropy_epoch)