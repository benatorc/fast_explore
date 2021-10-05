from collections import defaultdict
import random
import numpy as np
from rdkit import DataStructs
import math

import mol_explore.eval.eval_diversity as eval_utils
from mol_explore.utils.chem_utils import combine_smiles, delete_bond
from mol_explore.utils.fragment_utils import add_attachment, fragment_molecule, parse_smiles_attachment
from mol_explore.rl.rewards import MAX_HORIZON, prop_qualified
from mol_explore.rl.utils import avg_scores, split_scores

EPSILON = 1e-8
HIGH = 5.
LOW = -5.
MAX_NUM_ATOMS = 200
FRONTIER_SIZE = 100000
FRONTIER_BUFFER = FRONTIER_SIZE

class Frontier:
    """
    The frontier of chemical space exploration. contains the fields:
      discovered: found qualified molecule set, satisfying diversity criterion.
      init_basis: base molecules to start search on.
      sim_func, sim_threshold: similarity comparision criterion and threshold.
      gold, gold_fps: given gold molecule set for novelty computation.
      TODO: need a mechanism to not to reselect items already selected then removed. Maybe just keep all seen ones...
      keep growing the seen ones. But this might run into trouble later if this gets too big. right now
      use size 100000, which should be more than enough for all our experiments.
    """
    def __init__(self, init_basis, score_func):        
        self.n_base = len(init_basis)

        self.discovered = [None] * self.n_base  # Keeps track of best translation for each input
        self.init_basis = {
            'smiles': [x for x in init_basis], # hack to change from ConfigList to List
            'scores': list(score_func(init_basis)), # Score of the molecule in the init_basis
            'base_idx': list(range(self.n_base)), # links back to the initial molecule this was translated from 
            'founds': np.zeros([self.n_base]),
        }
        self.init_basis['fps'] = eval_utils.get_fps(self.init_basis['smiles'], fp_rad=2, fp_dim=2048)
    
    def update(self, discovered):

        # update init_basis
        self.init_basis['smiles'] = self.init_basis['smiles'] + list(discovered['smiles'])
        self.init_basis['scores'] = self.init_basis['scores'] + list(discovered['scores'])
        self.init_basis['base_idx'] = self.init_basis['base_idx'] + list(discovered['base_idx'])
        for x in discovered['base_idx']:
            self.init_basis['founds'][x] += 1

        # update set of discovered molecules 
        n_discovered = len(discovered['smiles'])
        for idx in range(n_discovered):
            base_idx = discovered['base_idx'][idx]
            score = discovered['scores'][idx]
            smiles = discovered['smiles'][idx]
            sim = discovered['sims'][idx]

            if self.discovered[base_idx] is None:
                self.discovered[base_idx] = (smiles, score, sim)
            else:
                old_score = self.discovered[base_idx][1]
                if score > old_score:
                    self.discovered[base_idx] = (smiles, score, sim)   

        
    @staticmethod
    def mean_list(x):
        return sum(x) / len(x)

        
class VecChemEnv:

    def __init__(self, init_basis, num_envs, frag_vae, score_func, 
                 end_criterion, golds, fp_type, prop_threshold, sim_threshold,
                 frontier_factor, prop_factor, relaxed_reward):
        self.frontier = Frontier(init_basis, score_func)
        self.num_envs = num_envs
        self.frag_vae = frag_vae
        self.envs = [ChemEnv(prop_factor=prop_factor) for _ in range(num_envs)]
        self.score_func = score_func
        self.end_criterion = end_criterion
        self.fp_type = fp_type
        self.prop_threshold = prop_threshold
        self.sim_threshold = sim_threshold
        self.frontier_factor = frontier_factor
        self.prop_factor = prop_factor
        self.relaxed_reward = relaxed_reward

    def reset(self):
        obs = [env.reset(self.frontier.init_basis) for env in self.envs]
        return np.array(obs)

    def step(self, actions, frag_list=None):
        action_types = actions['type'].detach().cpu().numpy()
        action_delete = actions['deletion'].detach().cpu().numpy()
        action_attach = actions['attachment'].detach().cpu().numpy()
        action_frag = actions['fragment']

        if frag_list is None:
            action_frag = self.frag_vae.decode_action(action_frag)['smiles']
        else:
            action_frag = [frag_list[x] for x in action_frag]
        outputs = [self.envs[i].exec_action(int(a_t), int(a_d), int(a_a), a_f)
                   for (i, (a_t, a_d, a_a, a_f)) in 
                   enumerate(zip(action_types, action_delete, action_attach, action_frag))]
        obs, new_obs, failed_transition, base_idx = list(map(np.array, zip(*outputs)))
        env_steps = np.array([self.envs[i].env_step for i in range(self.num_envs)])
        
        ob_fps = eval_utils.get_fps(new_obs, fp_type=self.fp_type)
        all_obs = np.concatenate([obs, new_obs])
        scores = self.score_func(all_obs)
        old_scores, new_scores = scores[:self.num_envs], scores[self.num_envs:]

        ob_fps = eval_utils.get_fps(new_obs, fp_type=self.fp_type, fp_rad=2, fp_dim=2048)
        done = self.end_criterion(env_steps, new_scores, failed_transition)
       
        base_scores = [self.frontier.init_basis['scores'][x] for x in base_idx]
        base_fps = [self.frontier.init_basis['fps'][x] for x in base_idx]

        base_sim = []
        for env_idx in range(len(base_fps)):
            sim = DataStructs.TanimotoSimilarity(ob_fps[env_idx], base_fps[env_idx])
            base_sim.append(sim)
        base_sim = np.array(base_sim)

        qualified_sim = base_sim >= self.sim_threshold
        qualified_prop = (new_scores - base_scores) > 0
        qualified = np.logical_and(qualified_prop, qualified_sim)
        qualified = np.logical_and(qualified, np.logical_not(failed_transition))

        reward = new_scores - old_scores
        if not self.relaxed_reward:
            reward = reward * qualified_sim.astype(float)
        reward[failed_transition] = -1  # penalize invalid transitions -- maybe not doing this too harshly? action may get too limited?
        
        # NOTE: logging purpose. can disable later with verbose flag.
        env_actions = []
        env_scores_old, env_scores_new = [], []
        for env_idx in range(self.num_envs):
            env_scores_old.append({'score': old_scores[env_idx]})
            env_scores_new.append({'score': new_scores[env_idx]})
            env_actions.append({
                'action_type': action_types[env_idx],
                'delete_idx': action_delete[env_idx],
                'delete_probs': actions['deletion_probs'][env_idx],
                'attach_idx': action_attach[env_idx],
                'attach_probs': actions['attach_probs'][env_idx],
            })

        outputs = [self.envs[i].after_exec(o1, o2, reward, s1, s2, action, done, self.frontier.init_basis)  # NOTE: if only start from init basis, remove the current here
                for (i, (o1, o2, reward, s1, s2, action, done)) in
                enumerate(zip(obs, new_obs, reward, env_scores_old, env_scores_new, env_actions, done))]
        final_obs, info = list(map(np.array, zip(*outputs)))

        self.frontier.init_basis['founds'][base_idx[qualified]] += 1
        discovered = {
            'scores': new_scores[qualified],
            'base_idx': base_idx[qualified],
            'smiles': new_obs[qualified],
            'sims': base_sim[qualified],
            'fps': [ob_fps[i] for i in range(len(qualified)) if qualified[i]], }

        self.frontier.update(discovered)

        return final_obs, reward, done, info

class ChemEnv:
    """
    score_func: a function that evaluates the quality of a molecule (SMILES string).
    end_criterion: a function that, given a molecule (SMILES string), decides whether this generation episode should end.
    """
    def __init__(self, prop_factor=1.0):
        self.prop_factor = prop_factor

    def reset(self, init_basis):
        if random.random() > .5:
            sample_idx = int(np.random.choice(np.arange(len(init_basis['smiles']))))
            self.base = init_basis['smiles'][sample_idx]
        else:
            unfinished_base_idx = []
            for idx, n in enumerate(init_basis['founds']):
                if n == 0:
                    unfinished_base_idx.append(idx)
            n_basis = len(init_basis['fps'])
            n_unfinished = len(unfinished_base_idx)
            n_finished = n_basis - n_unfinished
            if (n_unfinished == 0) or (n_finished >= n_basis * self.prop_factor):
                unfinished_base_idx = np.arange(len(init_basis['fps']))
            sample_idx = int(np.random.choice(unfinished_base_idx))
            self.base = init_basis['smiles'][sample_idx]
            # try:
            #     sample_base, _ = fragment_molecule(init_basis['smiles'][sample_idx])
            #     sample_base, _ = parse_smiles_attachment(sample_base)
            #     if sample_base is not None:
            #         self.base = sample_base
            # except:
            #     pass
        self.base_idx = init_basis['base_idx'][sample_idx]
        self.env_step = 0
        self.reward = 0

        return self.base
    
    def exec_action(self, action_type, action_delete, action_attach, action_frag):
        if action_type == 0:
            try:
                base_marked = add_attachment(self.base, action_attach)
                obs = combine_smiles(base_marked, action_frag)
                if obs is None:
                    assert False
                failed_transition = False
            except:
                print('Invalid addition')
                obs = self.base
                failed_transition = True
            
        elif action_type == 1:
            try:
                obs = delete_bond(self.base, action_delete)
                failed_transition = False
            except:
                print('Invalid deletion')
                obs = self.base
                failed_transition = True
        return self.base, obs, failed_transition, self.base_idx

    def after_exec(self, obs_old, obs_new, reward, scores_old, scores_new, actions, done, init_basis):
        self.env_step += 1
        self.base = obs_new
        self.reward += reward
                
        info = {'step': self.env_step, 'reward': float(self.reward),
                'done': done,
                'bad_transition': done and self.env_step > MAX_HORIZON,
                'obs_old': obs_old,
                'base': obs_new,
                'score_old': scores_old,
                'score': scores_new,
                'actions': actions}  # TODO MAX_HORIZON is hardcoded for now      
                
        if done:
            info['final'] = obs_new
            obs_new = self.reset(init_basis)
            
        return obs_new, info
