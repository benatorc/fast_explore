import pathlib
import os
import torch
import hydra
# import GPUtil
import numpy as np
import json
from collections import deque

import mol_explore.eval.eval_diversity as eval_utils
import mol_explore.rl.utils as utils
from mol_explore.models.mol_graph import MolGraph
from mol_explore.models.vocab import common_atom_vocab
from mol_explore.models.smiles_vae import make_cuda
from mol_explore.rl.logger import Logger
from mol_explore.rl.rewards import get_score_func, get_end_criterion, prop_qualified
from mol_explore.utils.model_utils import load_model
from mol_explore.utils.visualize_utils import draw_mol

FRONTIER_KEYS = ['smiles', 'fps']
        
ACTIVES_PATHS = {
    'gsk': 'data/gsk3/actives.txt',
    'jnk': 'data/jnk3/actives.txt',
    'gsk+jnk': 'data/gsk3_jnk3/actives.txt', 
    'zinc': 'data/zinc/zinc_input.json', }

class Workspace(object):
    def __init__(self, cfg, workdir=None):
        utils.set_seed_everywhere(cfg.seed)
        if workdir is None:
            self.work_dir = os.getcwd()
        else:
            self.work_dir = workdir
        
        self.log_dir = self.work_dir + '/logs'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        print(f'workspace: {self.work_dir}')
        self.work_dir = pathlib.Path(self.work_dir)

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.num_procs = cfg.num_procs
        self.batch_size = int(self.num_procs * cfg.storage.num_steps
                              / cfg.agent.num_mini_batch)
        print(f'batch_size={self.batch_size}')
        self.num_updates = int(cfg.num_train_steps // cfg.storage.num_steps
                               // self.num_procs)
        # log at every update
        self.steps_per_update = int(cfg.num_procs * cfg.storage.num_steps)
        self.num_mols = cfg.num_mols
        self.max_steps = cfg.max_steps

        # init frag encoder
        frag_vae = load_model(cfg.frag_vae.model_path)
        self.frag_vae = frag_vae
        self.frag_vae.train(False)
        self.cfg.frag_vae.latent_dim = frag_vae.latent_size
        self.cfg.frag_vae.n_embed = frag_vae.n_embed
        self.fp_type = cfg.env.fp_type
        self.action_type = cfg.actor.action_type

        if self.action_type == 'fixed':
            with open(cfg.fragment_actions, 'r') as fragment_file:
                self.frag_list = json.load(fragment_file)
        else:
            self.frag_list = None

        # init base encoder/agent
        encoder = hydra.utils.instantiate(cfg.encoder, avocab=common_atom_vocab)
        if self.frag_list is not None:
            actor = hydra.utils.instantiate(cfg.actor, encoder=encoder, action_dim=len(self.frag_list))
        else:
            actor = hydra.utils.instantiate(cfg.actor, encoder=encoder)
        critic = hydra.utils.instantiate(cfg.critic, encoder=encoder)
        self.agent = hydra.utils.instantiate(cfg.agent, encoder=encoder, actor=actor, critic=critic)
        self.rollouts = hydra.utils.instantiate(cfg.storage)
        self.rollouts.to(self.cfg.device)
        
        # rl explore
        if self.cfg.init_basis.endswith('.json'):
            with open(self.cfg.init_basis, 'r') as f:
                init_basis = json.load(f)
                self.cfg.init_basis = init_basis
        elif self.cfg.init_basis.endswith('.txt'): 
            with open(self.cfg.init_basis, 'r') as f:
                lines = [line.rstrip() for line in f]
                init_basis = [line.split(',')[0] for line in lines[1:]]
                self.cfg.init_basis = init_basis
        else:
            raise NotImplementedError
        
        actives_path = None
        if 'gsk' in self.cfg.task:
            actives_path = ACTIVES_PATHS['gsk']
        if 'jnk' in self.cfg.task:
            actives_path = ACTIVES_PATHS['jnk']
        if 'gsk' in self.cfg.task and 'jnk' in cfg.task:
            actives_path = ACTIVES_PATHS['gsk+jnk']
        if actives_path is None:
            # protein dataset, hack
            actives_path = ACTIVES_PATHS[self.cfg.task]

        with open(actives_path, 'r') as f:
            if actives_path.endswith('txt'):
                mylist = f.read().splitlines()
                self.golds = [s.split(',')[0] for s in mylist[1:]]
            elif actives_path.endswith('json'):
                self.golds = json.load(f)
            self.gold_fps = eval_utils.get_fps(self.golds, fp_type=self.fp_type)

        if 'gsk' in self.cfg.task or 'jnk' in self.cfg.task:
            env_type = 'ligand'
        elif 'zinc' in self.cfg.task:
            env_type = 'zinc'
        self.env_type = env_type

        score_func = get_score_func(cfg.task)
        end_criterion = get_end_criterion(env_type)

        env_kwargs = { 
            "init_basis": init_basis,
            "frag_vae": frag_vae,
            "score_func": score_func,
            "end_criterion": end_criterion,
            "golds": (self.golds, self.gold_fps),
        }
        if 'gsk' in self.cfg.task or 'jnk' in self.cfg.task:
            self.envs = hydra.utils.instantiate(cfg.env, **env_kwargs)
        else:
            self.envs = hydra.utils.instantiate(cfg.env_reg, **env_kwargs)
        self.score_func = score_func
        self.logger = Logger(self.work_dir, 0, 'metrics.jsonl')

        # logging
        self.update = 0
        self.step = 0

    def train(self):
        num_episodes_per_proc = np.zeros(self.num_procs, dtype=int)
        train_episode_lengths = deque(maxlen=100)
        train_episode_rewards = deque(maxlen=100)

        obs = self.envs.reset()
        np.copyto(self.rollouts.obses[0], obs)
        for update in range(self.num_updates):
            all_infos = []
            for step_in_update in range(self.rollouts.num_steps):
                with torch.no_grad():
                    obs = self.rollouts.obses[step_in_update]
                    obs_graph, masks = MolGraph.tensorize_action(obs, avocab=common_atom_vocab)
                    obs_graph = make_cuda(obs_graph)
                    add_mask, del_mask, edge_scope = masks
                    obs_tensor = {'base': obs_graph, 'add_mask': add_mask.cuda(), 
                                  'del_mask': del_mask.cuda(), 'edge_scope': edge_scope}
                    action = self.agent.act(obs_tensor)
                    value, action_log_prob, dist_entropy = self.agent.ac_evaluate(
                        obs_tensor, action)
                obs, reward, done, infos = self.envs.step(action, frag_list=self.frag_list)
                if self.cfg.verbose:
                    all_infos.append(infos)
                self.step += self.cfg.num_procs

                for proc_idx, info in enumerate(infos):
                    if info['done']:
                        num_episodes_per_proc[proc_idx] += 1
                        train_episode_lengths.append(info['step'])
                        train_episode_rewards.append(info['reward'])

                reward = torch.FloatTensor(reward).view(-1, 1)
                mask = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                # TODO this may need to be added to the Env, figure out what this bad transition is
                bad_mask = torch.FloatTensor(
                    [[0.0] if info['bad_transition'] else [1.0]
                     for info in infos])
                self.rollouts.insert(obs, action, action_log_prob, value,
                                    reward, mask, bad_mask)

            with torch.no_grad():
                obs = self.rollouts.obses[-1]
                obs_graph, _ = MolGraph.tensorize_action(obs, avocab=common_atom_vocab)
                obs_graph = make_cuda(obs_graph)
                obs_tensor = {'base': obs_graph}
                next_value = self.agent.get_value(obs_tensor)

            self.rollouts.compute_returns(next_value, self.cfg.gamma, self.cfg.gae_lambda)
            self.agent.update(self.rollouts, self.logger, self.step)
            self.rollouts.after_update()
            self.update = update

            # logging
            self.logger.scalar('train/update', self.update)
            self.logger.scalar('train/episode', num_episodes_per_proc.sum())
            self.logger.scalar('train/episode_length', np.mean(train_episode_lengths))
            self.logger.scalar('train/episode_reward', np.mean(train_episode_rewards))
            self.logger.scalar('train/reward_std', np.std(train_episode_rewards))
            self.logger.scalar('train/init_basis_size', len(self.envs.frontier.init_basis['smiles']))
            self.logger.scalar('train/addition_rate', 1-self.rollouts.actions_type.sum()/self.steps_per_update)

            if self.env_type == 'ligand':
                self.logger.scalar('train/frontier_size', len(self.envs.frontier.discovered['smiles']))

            if self.env_type == 'zinc':
                n_unmatched, n_matched, scores_sum, sim_sum = 0, 0, 0, 0, 
                for idx, x in enumerate(self.envs.frontier.discovered):
                    if x is None:
                        n_unmatched += 1
                    else:
                        n_matched += 1
                        scores_sum += x[1] - self.envs.frontier.init_basis['scores'][idx]
                        sim_sum += x[2]
                
                # edge case
                if n_matched == 0:
                    n_matched = 1e-6
                avg_score = scores_sum / n_matched
                avg_sim = sim_sum / n_matched

                self.logger.scalar('train/n_unmatched', n_unmatched)
                self.logger.scalar('train/avg_score', avg_score)
                self.logger.scalar('train/avg_sim', avg_sim)

            self.logger.step = self.step
            self.logger.write(fps=True)

            if self.update % self.cfg.eval_frequency == 0:
                if self.env_type == 'ligand' and len(self.envs.frontier.discovered['smiles']) > 0:
                    self.log_images()
                self.agent.save_checkpoint(self.work_dir / 'checkpoint.pt', self.step)
            
            if self.cfg.verbose:
                np.save('%s/train_%d.npy' % (self.log_dir, self.update), all_infos)
            
            # end a search when reach the target num molecules
            if self.env_type =='ligand' and len(self.envs.frontier.discovered['smiles']) >= self.num_mols:
                np.save('frontier.npy', self.envs.frontier)
                sr, diversity, novelty = self.evaluate()
                self.logger.scalar('eval/SR', sr)
                self.logger.scalar('eval/DIV', diversity)
                self.logger.scalar('eval/NOV', novelty)
                self.logger.write()

                summary_path = self.work_dir._str + '/summary.json'
                with open(summary_path, 'w') as summary_file:
                    json.dump({
                        'sr': sr, 
                        'nov': novelty, 
                        'div': diversity, 
                        'time': self.logger.get_total_time()
                    }, summary_file)

                print('Training complete, num molecules: %d, Sucess: %.3f, Novelty: %.3f, Diversity: %.3f' % (
                    len(self.envs.frontier.discovered['smiles']), sr, novelty, diversity))
                return

            if self.env_type == 'zinc':
                np.save('frontier.npy', self.envs.frontier)
            
            if self.max_steps != -1 and self.step > self.max_steps:
                return

    def log_images(self):
        samples = np.random.choice(self.envs.frontier.discovered['smiles'], 3)        
        mol_image = draw_mol(samples)
        self.logger.image('train/mol_image', mol_image)
        self.logger.write(fps=False)
        
    def log_histograms(self):
        for s_name in self.envs.frontier.discovered.keys():
            if s_name not in FRONTIER_KEYS:
                self.logger.histogram(f'train/frontier_score_{s_name}', self.envs.frontier.discovered[s_name])
                
    def evaluate(self):
        proposal = self.envs.frontier.discovered['smiles'][:self.num_mols]
        proposal_fps = eval_utils.get_fps(list(proposal), fp_type=self.fp_type)
        scores = self.score_func(proposal)
        success_rate = prop_qualified(scores).mean()
        diversity = eval_utils.compute_diversity(proposal_fps)
        novelty = eval_utils.compute_novelty(self.gold_fps, proposal_fps)
        return success_rate, diversity, novelty

@hydra.main(config_name='config')
def main(cfg):
    workspace = Workspace(cfg)
    workspace.train()


if __name__ == "__main__":
    main(None)
