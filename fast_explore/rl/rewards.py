# functions for compute property scores, exploration bonus, etc.
import numpy as np
from functools import partial
from mol_explore.scorers import (
        gsk3_rf_model, jnk3_rf_model, gsk3_qed_rf_model, gsk3_qed_sa_rf_model,
        jnk3_qed_sa_rf_model, gsk3_jnk3_model, gsk3_jnk3_qed_sa_model, constrained_logP_model)

MAX_HORIZON = 10

TASK2SF = {
    'gsk': partial(gsk3_rf_model),
    'jnk': partial(jnk3_rf_model),
    'gsk+jnk': partial(gsk3_jnk3_model),
    'gsk+qed': partial(gsk3_qed_rf_model),
    'gsk+qed+sa': partial(gsk3_qed_sa_rf_model),
    'jnk+qed+sa': partial(jnk3_qed_sa_rf_model),
    'gsk+jnk+qed+sa': partial(gsk3_jnk3_qed_sa_model),
    'zinc': partial(constrained_logP_model),
}

# Thresholds are >= equalities (good molecules must exceed this threshold)
SCORE_THRESHOLDS = {
    'gsk': .5,
    'jnk': .5,
    'sa': .6,
    'qed': .6, }

def get_score_func(task='gsk'):
    sf = TASK2SF[task]
    if 'gsk' in task or 'jnk' in task:
        return sf()
    elif 'zinc' in task:
        return sf()
    else:
        return sf(task)

def prop_qualified(score_dict, prop_factor=1.0):
    qualified = None
    for name, scores in score_dict.items():
        if qualified is None:
            qualified = np.ones_like(scores)  # Initialize

        assert name in SCORE_THRESHOLDS
        scores_qualified = scores >= SCORE_THRESHOLDS[name] * prop_factor
        qualified = np.logical_and(qualified, scores_qualified)
    return qualified

def naive_end_criterion(step, score, failed_transition):
    # End if the max number of steps have been reached, or the property threshold is not met
    max_step_reached = step > MAX_HORIZON
    
    property_gsk3_end, property_jnk3_end = None, None
    if 'gsk' in score:
        property_gsk3_end = score['gsk'] < SCORE_THRESHOLDS['gsk'] 
        # * np.tanh(3 * step / MAX_HORIZON)  
        # * (step / MAX_HORIZON) # linear increasing score requirement
        # * np.tanh(3 * step / MAX_HORIZON) # tanh increasing score requirement
    if 'jnk' in score:
        property_jnk3_end = score['jnk'] < SCORE_THRESHOLDS['jnk']
    assert property_gsk3_end is not None or property_jnk3_end is not None

    if property_gsk3_end is None:
        property_end = property_jnk3_end
    elif property_jnk3_end is None:
        property_end = property_gsk3_end
    else:
        # Both property has to be below threshold to end if using dual properties
        property_end = np.logical_and(property_jnk3_end, property_gsk3_end)

    normal_dones = np.logical_or(max_step_reached, property_end)  # TODO the score requirement can be increasing with the step. allow intermediate steps to have low score
    return np.logical_or(normal_dones, failed_transition)

def zinc_end_criterion(step, score, failed_transition):
    max_step_reached = step > MAX_HORIZON

    return np.logical_or(failed_transition, max_step_reached)

def get_end_criterion(env_type=None):
    if env_type is None:
        return naive_end_criterion
    
    if env_type == 'ligand':
        return naive_end_criterion
    elif env_type == 'zinc':
        return zinc_end_criterion
    else:
        print('Invalid env type: %s' % env_type)
        assert False

def new_func():
    return None
