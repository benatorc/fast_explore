import pickle
import numpy as np
import torch

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class ChemBuffer(object):
    def __init__(self, action_dim, num_steps, num_procs, action_type):
        self.action_type = action_type
        self.action_dim = action_dim
        if action_type == 'fixed':
            self.action_dim = 1
        self.obses = np.empty([num_steps + 1, num_procs], dtype=object)
        # save numbers as tensors
        self.rewards = torch.zeros([num_steps, num_procs, 1])
        self.actions_type = torch.zeros([num_steps, num_procs, 1]).long()
        self.actions_del = torch.zeros([num_steps, num_procs, 1]).long()
        self.actions_attach = torch.zeros([num_steps, num_procs, 1]).long()
        if action_type == 'discrete':
            self.actions_frag = torch.zeros([num_steps, num_procs, action_dim, 1])  # last <1> for discrete action
        elif action_type == 'fixed':
            self.actions_frag = torch.zeros([num_steps, num_procs, 1])
        else:
            assert False
        self.action_log_probs = torch.zeros([num_steps, num_procs, 1])
        self.value_preds = torch.zeros([num_steps + 1, num_procs, 1])
        self.returns = torch.zeros([num_steps + 1, num_procs, 1])
        self.masks = torch.zeros([num_steps + 1, num_procs, 1])

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.zeros([num_steps + 1, num_procs, 1])

        self.num_steps = num_steps
        self.idx = 0
    
    def to(self, device):
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions_type = self.actions_type.to(device)
        self.actions_del = self.actions_del.to(device)
        self.actions_attach = self.actions_attach.to(device)
        self.actions_frag = self.actions_frag.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        
    def save(self, model_dir):
        with open('%s/buffer.pt' % (model_dir), 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def insert(self, obs, action, action_log_prob, 
               value, reward, mask, bad_mask):
        np.copyto(self.obses[self.idx+1], obs)
        self.rewards[self.idx].copy_(reward)
        self.actions_type[self.idx].copy_(action['type'])
        self.actions_del[self.idx].copy_(action['deletion'])
        self.actions_attach[self.idx].copy_(action['attachment'])
        self.actions_frag[self.idx].copy_(action['fragment'])
        self.action_log_probs[self.idx].copy_(action_log_prob)
        self.value_preds[self.idx].copy_(value)
        self.masks[self.idx+1].copy_(mask)
        self.bad_masks[self.idx+1].copy_(bad_mask)
        self.idx = (self.idx + 1) % self.num_steps

    def after_update(self):
        np.copyto(self.obses[0], (self.obses[-1]))
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        gamma,
                        gae_lambda):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            delta = self.rewards[step] + gamma * self.value_preds[
                step + 1] * self.masks[step + 1] - self.value_preds[step]
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            gae = gae * self.bad_masks[step + 1] 
            self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_procs = self.rewards.shape[0:2]
        batch_size = num_procs * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_procs, num_steps, num_procs * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)

        for indices in sampler:
            obs_batch = self.obses[:-1].reshape(-1, *self.obses.shape[2:])[indices]
            action_type_batch = self.actions_type.reshape(-1, self.actions_type.shape[-1])[indices]
            action_del_batch = self.actions_del.reshape(-1, self.actions_del.shape[-1])[indices]
            action_attach_batch = self.actions_attach.reshape(-1, self.actions_attach.shape[-1])[indices]
            action_frag_batch = self.actions_frag.reshape(-1, self.action_dim, 1)[indices]
            value_pred_batch = self.value_preds[:-1].reshape(-1, 1)[indices]
            return_batch = self.returns[:-1].reshape(-1, 1)[indices]
            mask_batch = self.masks[:-1].reshape(-1, 1)[indices]
            old_action_log_prob_batch = self.action_log_probs.reshape(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.reshape(-1, 1)[indices]
            
            action_batch = {
                'type': action_type_batch,
                'deletion': action_del_batch,
                'attachment': action_attach_batch,
                'fragment': action_frag_batch
            }

            yield obs_batch, action_batch, value_pred_batch, return_batch, mask_batch, old_action_log_prob_batch, adv_targ