import torch
import copy
import random
import numpy as np

def save_checkpoint(net, model_target, path):
    torch.save({
        'model_state_dict': net.state_dict(),
        'model_target_state_dict': model_target.state_dict(),
        }, path)

def copy_states(source, target):
    for key, _ in zip(source.state_dict()['state'].keys(), target.state_dict()['state'].keys()):

        target.state_dict()['state'][key]['exp_avg_sq'] = copy.deepcopy(source.state_dict()['state'][key]['exp_avg_sq'])
        target.state_dict()['state'][key]['exp_avg'] = copy.deepcopy(source.state_dict()['state'][key]['exp_avg'])
        target.state_dict()['state'][key]['step'] = copy.deepcopy(source.state_dict()['state'][key]['step'])
        
def target_model_ema(model, model_target, ema=0.995):
    with torch.no_grad():
        for param, param_target in zip(model.parameters(), model_target.parameters()):
            param_target.data = ema * param_target.data + (1.0 - ema) * param.data.clone()

# https://github.com/google/dopamine/blob/master/dopamine/jax/agents/dqn/dqn_agent.py
def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus

def epsilon_greedy(Q_action, n_actions, step, final_eps=0, num_envs=1):
    epsilon = linearly_decaying_epsilon(2001, step, 2000, final_eps)
    
    if random.random() < epsilon:
        action = torch.randint(0, n_actions, (num_envs,), dtype=torch.int64, device='cuda').squeeze(0)
    else:
        action = Q_action.view(num_envs).squeeze(0).to(torch.int64)
    return action

# https://github.com/google/dopamine/blob/master/dopamine/jax/agents/rainbow/rainbow_agent.py
def project_distribution(supports, weights, target_support):
    with torch.no_grad():
        v_min, v_max = target_support[0], target_support[-1]
        # `N` in Eq7.
        num_buckets = target_support.shape[-1]
        # delta_z = `\Delta z` in Eq7.
        delta_z = (v_max - v_min) / (num_buckets-1)
        # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
        clipped_support = supports.clip(v_min, v_max)
        # numerator = `|clipped_support - z_i|` in Eq7.
        numerator = (clipped_support[:,None] - target_support[None,:,None].repeat_interleave(clipped_support.shape[0],0)).abs()
        quotient = 1 - (numerator / delta_z)
        # clipped_quotient = `[1 - numerator / (\Delta z)]_0^1` in Eq7.
        clipped_quotient = quotient.clip(0, 1)
        # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))` in Eq7.
        inner_prod = (clipped_quotient * weights[:,None]).sum(-1)
        #inner_prod = (clipped_quotient).sum(-1) * weights
        return inner_prod.squeeze()

def params_and_grad_norm(model):
    param_norm, grad_norm = 0, 0
    for n, param in model.named_parameters():
        if not n.endswith('.bias'):
            param_norm += torch.norm(param.data)
            if param.grad is not None:
                grad_norm += torch.norm(param.grad)
    return param_norm, grad_norm