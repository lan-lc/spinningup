from ast import mod
from telnetlib import DM
from turtle import mode
from unicodedata import name
import gym
from copy import deepcopy
import os
import os.path as osp
import torch
from scipy import stats
from statistics import mean 
import numpy as np
from torch.optim import Adam
import itertools
import random
import torch.nn as nn
import argparse
import pickle

def get_env_name(name):
    if ('humanoid' in name) or ('Humanoid' in name):
        return 'Humanoid-v3'
    if ('halfcheetah' in name) or ('HalfCheetah' in name):
        return 'HalfCheetah-v3'
    if ('ant' in name) or ('Ant' in name):
        return 'Ant-v3'
    if ('hopper' in name) or  ('Hopper' in name) :
        return 'Hopper-v3'
    if ('walker' in name) or ('Walker' in name) :
        return 'Walker2d-v3'
    return 'unknown'

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class PPO_Actor():
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        self.pi = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.obs_mean = np.ones(obs_dim)
        self.obs_std = np.ones(obs_dim)
        self.clip = 10.0
        # print(type(self.pi))
    
    def normalize_o(self, o):
        o = o - self.obs_mean
        o = o / (self.obs_std + 1e-8)
        o = np.clip(o, -self.clip, self.clip)
        return o
    
    def act(self, o):
        o = self.normalize_o(o)
        o = torch.as_tensor(o, dtype=torch.float32)
        return self.pi(o).detach().numpy()
    
    def copy_model(self, md):
        self.pi.load_state_dict(md['pi'])
        self.obs_mean = md['obs_mean']
        self.obs_std = md['obs_std']
        self.clip = md['clip']
        
    def load(self, name):
        md = torch.load(name)
        self.copy_model(md)


def get_ppo_models(path, name):
    fpath = osp.join(path, name)
    models = []
    file_names = os.listdir(fpath)
    if len(file_names) == 0:
        return []
    env = gym.make(get_env_name(name))
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    for file_name in file_names:   
        if ".pt" not in file_name:
            continue
        fname = osp.join(fpath, file_name)
        print(file_name)
        model = PPO_Actor(obs_dim, action_dim, (64, 64), nn.Tanh)
        model.load(fname)
        models.append((name, file_name, model))
    return models

def get_models(path, name):
    if 'ppo' in name:
        return get_ppo_models(path, name)
    fpath = osp.join(path, name)
    models = []
    file_names = os.listdir(fpath)
    if len(file_names) == 0:
        return []
    for file_name in file_names:   
        # fname = osp.join(fpath, file_name ,'pyt_save', 'model0.pt')
        fname = osp.join(fpath, file_name ,'pyt_save', 'model.pt')
        print(fname)
        model = torch.load(fname)
        models.append((name, file_name, model))
    return models

def save_state(env):
    return env.sim.get_state()

def restore_state(env, old_state):
    env.reset()
    env.sim.set_state(old_state)
    env.sim.forward()
    return env.get_obs()

def get_ppo_action(o, md):
    return md.act(o)

def get_action(o, md, name):
    if 'ppo' in name:
        return get_ppo_action(o, md)
    if 'train' not in name:
        o = torch.as_tensor(o, dtype=torch.float32)
        return md.act(o)
    o = torch.as_tensor(o, dtype=torch.float32)
    return md.act(o, deterministic=False)

def test_model(env, model, num_episodes, name):
    max_ep_len = 1000
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    rets = []
    trajs = []
    all_rewards = []
    mid_points = []
    rewards = []
    while n < num_episodes:
        old_state = save_state(env)
        mid_points.append(old_state)
        a = get_action(o, model, name)
        o, r, d, _ = env.step(a)
        rewards.append(r)
        ep_ret += r
        ep_len += 1
        if d or (ep_len == max_ep_len):
            rets.append(ep_ret)
            trajs.append(mid_points)
            all_rewards.append(rewards)
            mid_points = []
            rewards = []
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1
    return rets, all_rewards, trajs

def print_rets(rets):
    rets = np.array(rets)
    print("mean, max, min, std", np.mean(rets),np.max(rets) , np.min(rets) , np.std(rets))
    return np.mean(rets)

def sample_trajs(trajs):
    sampled_trajs = []
    sample_num = 250
    for i in range(len(trajs)):
        tmp = []
        traj = trajs[i]
        l = len(traj) - 10
        if l - 10 > sample_num:
            ids = random.sample(range(1, l), sample_num)
            for id in ids:
                tmp.append((id, traj[id]))
        else:
            max_l = len(traj)
            if max_l > sample_num:
                max_l = sample_num
            for id in range(1, max_l):
                tmp.append((id, traj[id]))
        
        sampled_trajs.append(tmp)
    return sampled_trajs

def generate_trajs(path, name, num_episodes):
    env = gym.make(get_env_name(name))
    models = get_models(path, name)
    all_trajs = []
    avg_rets = []
    for model in models:
        print("start generating model ", model[0], model[1])
        rets, all_rewards, trajs = test_model(env, model[2], num_episodes, name)
        avg_ret = print_rets(rets)
        avg_rets.append(avg_ret)
        sampled_trajs = sample_trajs(trajs)
        all_trajs.append((model[0], model[1], rets, all_rewards, sampled_trajs))
    print_rets(avg_rets)
    traj_path = os.path.join(path, name + '_' + str(num_episodes) + "_trajs.pkl")
    print(traj_path)
    with open(traj_path , 'wb') as f:
        pickle.dump(all_trajs, f)

num_episodes = 1000
# path  = '/home/lclan/spinningup/data/'
path = '/home/lclan/spinningup/old_data'
parser = argparse.ArgumentParser()
parser.add_argument('--name_id', type=int, required = True)
# names = ['Ant-v3_sac_base' , 'Ant-v3_td3_base', 'atla_ppo_ant', 'vanilla_ppo_ant',
#         'Humanoid-v3_sac_base', 'Humanoid-v3_td3_base', 'vanilla_ppo_humanoid',  'sgld_ppo_humanoid',
#         'Walker2d-v3_sac_base', 'Walker2d-v3_td3_base', 'vanilla_ppo_walker', 'atla_ppo_walker',
#         'HalfCheetah-v3_sac_base', 'HalfCheetah-v3_td3_base',  'vanilla_ppo_halfcheetah', 'atla_ppo_halfcheetah',  
#         'Hopper-v3_sac_base', 'Hopper-v3_td3_base', 'vanilla_ppo_hopper',  'atla_ppo_hopper' ]
names = ['Ant-v3_sac_base_train', 'Humanoid-v3_sac_base_train', 'Walker2d-v3_sac_base_train', 'Hopper-v3_sac_base_train']
args =  parser.parse_args()
generate_trajs(path, names[args.name_id], num_episodes)
