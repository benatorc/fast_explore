from collections import defaultdict
import numpy as np

import mol_explore.eval.eval_diversity as eval_utils
from mol_explore.utils.chem_utils import combine_smiles, delete_bond
from mol_explore.utils.fragment_utils import add_attachment
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
    def __init__(self, init_basis, sim_func, sim_threshold, golds, fp_type):
        self.discovered = defaultdict(list)
        self.init_basis = {
            'smiles': [x for x in init_basis], # hack to change from ConfigList to List
            'visits': np.zeros(FRONTIER_SIZE+FRONTIER_BUFFER),  # Add buffer for overflow
            'founds': np.zeros(FRONTIER_SIZE+FRONTIER_BUFFER),
            'probs': np.zeros(FRONTIER_SIZE+FRONTIER_BUFFER)
        }
        self.init_basis['probs'][:len(init_basis)] = 1/len(init_basis)
        self.sim_func, self.sim_threshold = (sim_func, np.array(sim_threshold))
        assert self.sim_func in ['max', 'mean', 'mixed', 'mean_hard']
        self.gold, self.gold_fps = ([x for x in golds[0]], [x for x in golds[1]])  # hack to change from ConfigList to List
        self.fp_type = fp_type
    
    def update(self, discovered, milestones=None):
        for k in discovered.keys():
            self.discovered[k].extend(list(discovered[k]))

        # remove duplicates when adding to init_basis.
        new_basis = set(discovered['smiles']).difference(self.init_basis['smiles'])
        # mile stone init basis
        self.init_basis['smiles'] = self.init_basis['smiles'] + list(new_basis)
        if milestones is not None:
            new_basis = set(milestones).difference(self.init_basis['smiles'])
            self.init_basis['smiles'] = self.init_basis['smiles'] + list(new_basis)
        
        # compute new sampling probability using UCB
        avg_found = self.init_basis['founds'] / (self.init_basis['visits']+EPSILON)
        delta = np.sqrt(3/2 * np.log(self.init_basis['visits'].sum()+1) / (self.init_basis['visits']+EPSILON))
        ucb = avg_found + delta
        ucb[len(self.init_basis['smiles']):] = 0
        index_sort_by_prob = np.argsort(-ucb)  # ucb index from largest to smallest
        try:
            self.init_basis = {k: v[index_sort_by_prob] if isinstance(v, np.ndarray) 
                           else list(np.array(v)[index_sort_by_prob[:len(self.init_basis['smiles'])]])
                           for k, v in self.init_basis.items()}
        except:
            import pdb; pdb.set_trace()
        
        # pop lowest prob basis out when init_basis exceeds size limit.
        for k in self.init_basis.keys():
            if isinstance(self.init_basis[k], np.ndarray):
                self.init_basis[k][FRONTIER_SIZE:] = 0
            else:
                self.init_basis['smiles'] = self.init_basis['smiles'][:FRONTIER_SIZE]
        
        ucb = ucb[index_sort_by_prob]
        ucb[FRONTIER_SIZE:] = 0
        probs = ucb / ucb.sum()
        probs[len(self.init_basis['smiles'])-1] = 1. - probs[:len(self.init_basis['smiles'])-1].sum()  # correct numerical error
        self.init_basis['probs'] = probs

    @staticmethod
    def mean_list(x):
        return sum(x) / len(x)

    def qual_dupe(self, obs):
        qualified_dupe = []
        cur_set = set(self.discovered['smiles'] + self.gold)
        for i in range(len(obs)):
            if obs[i] not in cur_set:
                qualified_dupe.append(True)
                # also remove dups in the same batch.
                cur_set = cur_set.union([obs[i]])
            else:
                qualified_dupe.append(False)
        return np.array(qualified_dupe)

    def qual_sim(self, obs, qualified, threshold):
        # If max, compare to the frontier
        # If mean, compare to the frontier + actives
        # If mixed, compare to frontier with mean, compare to actives with max
        if self.sim_func == 'max':
            get_sim = lambda ob, qual: max(eval_utils.get_sims_from_fps(eval_utils.get_fps([ob], fp_type=self.fp_type)[0],
                                           list(self.discovered['fps']))) if (qual and len(self.discovered['smiles']) > 0) else 0 
        elif self.sim_func == 'mean':
            get_sim = lambda ob, qual: Frontier.mean_list(eval_utils.get_sims_from_fps(eval_utils.get_fps([ob], fp_type=self.fp_type)[0],
                                                          list(self.discovered['fps'] + self.gold_fps))) if qual else 0
        elif self.sim_func == 'mean_hard':
            def get_sim(ob, qual):
                if not qual:
                    return 1, 1
                obs_fps = eval_utils.get_fps([ob], fp_type=self.fp_type)[0]
                actives_sim = Frontier.mean_list(eval_utils.get_sims_from_fps(obs_fps, self.gold_fps))
                if len(self.discovered['smiles']) > 0:
                    frontier_sim = Frontier.mean_list(eval_utils.get_sims_from_fps(obs_fps, list(self.discovered['fps'])))
                else:
                    frontier_sim = 0
                return actives_sim, frontier_sim
        else:
            def get_sim(ob, qual):
                if not qual:
                    return 1, 1  # fails similarity check automatically if not qual for prop.
                ob_fps = eval_utils.get_fps([ob], fp_type=self.fp_type)[0]
                actives_sim = max(eval_utils.get_sims_from_fps(ob_fps, self.gold_fps))
                if len(self.discovered['smiles']) > 0:
                    frontier_sim = Frontier.mean_list(eval_utils.get_sims_from_fps(ob_fps, list(self.discovered['fps'])))
                else:
                    frontier_sim = 0
                return actives_sim, frontier_sim

        similarity = np.array([get_sim(ob, qual) for ob, qual in zip(obs, qualified)])  # this risk adding two similar ones in the same batch       
        if self.sim_func in ['max', 'mean']:
            qualified_sim = similarity < threshold
        elif self.sim_func == 'mean_hard':
            qualified_sim = np.logical_and(similarity[:, 0] < threshold[0], similarity[:, 1] < threshold[1])
        else:
            # sim_func = mixed, hard code 0.4 for actives sim
            qualified_sim = np.logical_and(similarity[:, 0] < threshold[0], similarity[:, 1] < threshold[1])
        return qualified_sim

class VecChemEnv:

    def __init__(self, init_basis, num_envs, frag_vae, score_func, 
                 end_criterion, golds, sim_threshold, sim_func, fp_type, 
                 frontier_factor, prop_factor, use_ucb_scores):
        self.frontier = Frontier(init_basis, sim_func, sim_threshold, golds, fp_type)
        self.num_envs = num_envs
        self.frag_vae = frag_vae
        self.envs = [ChemEnv() for _ in range(num_envs)]
        self.score_func = score_func
        self.end_criterion = end_criterion
        self.fp_type = fp_type
        self.frontier_factor = frontier_factor
        self.prop_factor = prop_factor
        self.use_ucb_scores = use_ucb_scores

    def reset(self):
        obs = [env.reset(self.frontier.init_basis, random=not self.use_ucb_scores) for env in self.envs]
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
        scores = self.score_func(np.concatenate([obs, new_obs]))
        new_score_dicts, old_score_dicts = split_scores(scores, self.num_envs)
        done = self.end_criterion(env_steps, new_score_dicts, failed_transition)
        reward = np.zeros(self.num_envs)
        qualified_prop = prop_qualified(new_score_dicts)
        try:
            ob_fps = eval_utils.get_fps(new_obs, fp_type=self.fp_type)
        except:
            import pdb; pdb.set_trace()

        qualified_dupe = self.frontier.qual_dupe(new_obs)
        qualified_sim = self.frontier.qual_sim(new_obs, qualified_prop, self.frontier.sim_threshold)
        qualified = np.logical_and(qualified_prop, qualified_sim)  # TODO 0.4 and 0.5 can be moving target
        qualified = np.logical_and(qualified, qualified_dupe)
        
        # milestones
        milestone_props = prop_qualified(new_score_dicts, prop_factor=self.prop_factor)
        qualified_milestones = self.frontier.qual_sim(new_obs, milestone_props, self.frontier.sim_threshold * self.frontier_factor)
        qualified_milestones = np.logical_and(milestone_props, qualified_milestones)
        
        # should get separate reward for prop qualification / diversity qualification.
        # reward[qualified_prop] += 0.1
        reward[qualified] += 1.  # reward if found novel molecule. TODO: this might not be the best reward. a novel molecule at this round would not be so next round.
        reward[failed_transition] += -0.1  # penalize invalid transitions -- maybe not doing this too harshly? action may get too limited?
        
        # NOTE: logging purpose. can disable later with verbose flag.
        env_actions = []
        env_scores_old, env_scores_new = [], []
        for env_idx in range(self.num_envs):
            env_scores_old.append({k: float(old_score_dicts[k][env_idx]) for k, _ in old_score_dicts.items()})
            env_scores_new.append({k: float(new_score_dicts[k][env_idx]) for k, _ in new_score_dicts.items()})
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
        
        # update self.frontier
        self.frontier.init_basis['founds'][base_idx[qualified]] += 1
        discovered = {k: v[qualified] for k, v in new_score_dicts.items()}
        discovered.update({
            'smiles': new_obs[qualified], 
            'fps': [ob_fps[i] for i in range(len(qualified)) if qualified[i]],
        })
        milestones = new_obs[qualified_milestones]
        self.frontier.update(discovered, milestones)
        
        try:
            discovered_smiles = self.frontier.discovered['smiles']
            if len(discovered_smiles) != len(set(discovered_smiles)):
                import pdb; pdb.set_trace()
        except:
            import pdb; pdb.set_trace()

        return final_obs, reward, done, info

class ChemEnv:
    """
    score_func: a function that evaluates the quality of a molecule (SMILES string).
    end_criterion: a function that, given a molecule (SMILES string), decides whether this generation episode should end.
    """

    def reset(self, init_basis, random=False):
        # resample if the base has no available attachment, though this would be weird to happen
        try:
            if random:
                base_idx = np.random.choice(np.arange(len(init_basis['smiles'])))
            else:
                base_idx = np.random.choice(np.arange(len(init_basis['smiles'])), 
                                        p=init_basis['probs'][:len(init_basis['smiles'])])
        except:
            import pdb; pdb.set_trace()
        self.base = init_basis['smiles'][int(base_idx)]
        self.base_idx = base_idx
        self.env_step = 0
        self.reward = 0
        # update visit counts.
        init_basis['visits'][base_idx] += 1
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
