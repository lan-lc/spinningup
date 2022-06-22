from ast import mod
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

def print_2f(*args):
    __builtins__.print(*("%.2f" % a if isinstance(a, float) else a
                         for a in args))

def get_action(o, md):
    o = torch.as_tensor(o, dtype=torch.float32)
    a = md.act(o)
    return a

def test_model(env, model, max_ep_len=None, num_episodes=20, interval = 1):
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    total_rewards = []
    trajs = []
    mid_points = []
    ds = []
    while n < num_episodes:
        if ep_len % interval == 0:
            old_state = deepcopy(env.sim.get_state())
            mid_points.append((ep_len, old_state, o))
        a = get_action(o, model)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        if d or (ep_len == max_ep_len):
            total_rewards.append(ep_ret)
            trajs.append(mid_points)
            mid_points = []
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1
    return total_rewards, trajs


def get_models(path, env_name, name):
    env = gym.make(env_name)
    fpath = osp.join(path, name)
    print(fpath)
    models = []
    while 1:
        fname = osp.join(fpath, name + "_s" + str(len(models)) ,'pyt_save', 'model.pt')
        print(fname)
        if(osp.exists(fname)):
            model = torch.load(fname)
            models.append(model)
        else:
            break
    # print(len(models))
    test_val = []
    for model in models:
        x, _ = test_model(env, model, num_episodes=10)
        # print(x, stats.trim_mean(x, 0.1))
        test_val.append(stats.trim_mean(x, 0.1))
    print_2f(test_val)
    sorted_ids = np.argsort(test_val)
    model1 = models[sorted_ids[-1]]
    model2 = models[sorted_ids[-2]]
    return model1, model2

def get_ppo_models(path, env_name, name):
    env = gym.make(env_name)
    fpath = osp.join(path, name)
    print(fpath)
    models = []
    while 1:
        fname = osp.join(fpath, name + "_" + str(len(models)) + '.pt')
        print(fname)
        if(osp.exists(fname)):
            env = gym.make(env_name)
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            model = PPO_Actor(obs_dim, action_dim, (64, 64), nn.Tanh)
            model.load(fname)
            models.append(model)
        else:
            break
    # print(len(models))
    test_val = []
    for model in models:
        x, _ = test_model(env, model, num_episodes=10)
        # print(x, stats.trim_mean(x, 0.1))
        test_val.append(stats.trim_mean(x, 0.1))
    
    sorted_ids = np.argsort(test_val)
    model1 = models[sorted_ids[-1]]
    model2 = models[sorted_ids[-2]]
    test_val.sort()
    print_2f(test_val)
    return model1, model2


def run_extra_steps(env, o, ep_len, md, max_ep_len, step_num = 50):
    # return 0
    # print(env.done)
    total_r = 0
    for i in range(step_num):
        a = get_action(o, md)
        o, r, d, _ = env.step(a)
        total_r += r
        ep_len += 1
        if d or (ep_len == max_ep_len):
            # print(i, d, r, ep_len, max_ep_len)
            return (d, total_r)
    return (d, total_r)


def save_state(env):
    return env.sim.get_state()

def restore_state(env, old_state):
    env.sim.set_state(old_state)
    env.sim.forward()
    return env.get_obs()

def vmean(v):
    return sum(v) / len(v)

def print_vpercent(v, sp = [0.25, 0.5, 1, 2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 40, 45, 50, 75, 99]):
    l = len(v)
    print_2f(*sp)
    vs = [v[int(l*x/100)] for x in sp]
    print_2f(*vs)


def test_intial_state(path, env_name, name, noise_scales):
    print(noise_scales)
    model1, model2 = get_models(path, env_name, name)
    for ns in noise_scales:
        print("ns: ",  ns)
        env = gym.make(env_name, reset_noise_scale=ns)
        r1 = []
        d1 = []
        for _ in range(2000):
            o = env.reset()
            # state = save_state(env)
            dd, rr = run_extra_steps(env, o, 0, model1, 1000, 998)
            if(rr < 100):
                dd = True
            r1.append(rr)
            d1.append(dd)
            # o = restore_state(env, state)
            # dd, rr = run_extra_steps(env, o, 0, model2, 1000, 998)
            # if(rr < 100):
            #     dd = True
            # r2.append(rr)
            # d2.append(dd)
        
        # print_2f("average and min m1 m2: ", vmean(r1), vmean(r1), min(r1), min(r2))
        # print_2f("done m1, m2",vmean(d1), vmean(d2))     
        # total = len(d1)
        # cnt = [0,0,0,0]
        # for i in range(total):
        #     c = 0
        #     if d1[i]:
        #         c+=1
        #     if d2[i]:
        #         c+=2
        #     cnt[c] += 100/(total)
        # print_2f("00, 01, 10, 11 ", cnt, (cnt[1]+cnt[3])*(cnt[2]+cnt[3])/100 )
        rs = deepcopy(r1) 
        rs.sort()
        print('total and mean: ', len(rs), vmean(rs), vmean(d1))
        print_2f(".25 .5 1  2.5  5  7.5  10  12.5  15  20  25  30  35  40  45  50  75  100 ")
        print_2f(rs[5], rs[10], rs[20], rs[50], rs[100], rs[150], rs[200], rs[250], rs[300], rs[400], rs[500], rs[600], rs[700], rs[800], rs[900], rs[1000], rs[1500], rs[1999])


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
        print(type(self.pi))
    
    def normalize_o(self, o):
        o = o - self.obs_mean
        o = o / (self.obs_std + 1e-8)
        o = np.clip(o, -self.clip, self.clip)
        return o
    
    def act(self, o):
        if torch.is_tensor(o):
            o = o.numpy()
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


def test_ppo_intial_state(env_name, name, noise_scales):
    print(noise_scales, env_name)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model1 = PPO_Actor(obs_dim, action_dim, (64, 64), nn.Tanh)
    model1.load(name)
    for ns in noise_scales:
        print("ns: ",  ns)
        env = gym.make(env_name, reset_noise_scale=ns)
        r1 = []
        d1 = []
        for _ in range(2000):
            o = env.reset()
            dd, rr = run_extra_steps(env, o, 0, model1, 1000, 998)
            if(rr < 100):
                dd = True
            r1.append(rr)
            d1.append(dd)
        rs = deepcopy(r1) 
        rs.sort()
        print('total and mean: ', len(rs), vmean(rs), vmean(d1))
        # print_2f(".25 .5 1  2.5  5  7.5  10  12.5  15  20  25  30  35  40  45  50  75  100 ")
        # print_2f(rs[5], rs[10], rs[20], rs[50], rs[100], rs[150], rs[200], rs[250], rs[300], rs[400], rs[500], rs[600], rs[700], rs[800], rs[900], rs[1000], rs[1500], rs[1999])
        print_vpercent(rs)
        
def test_ppo_intial_state_2(path, env_name, name, noise_scales):
    print(path, noise_scales, name, env_name)
    model1, model2 = get_ppo_models(path, env_name, name)
    for ns in noise_scales:
        print("ns: ",  ns)
        env = gym.make(env_name, reset_noise_scale=ns)
        r1 = []
        d1 = []
        for _ in range(2000):
            o = env.reset()
            dd, rr = run_extra_steps(env, o, 0, model1, 1000, 998)
            if(rr < 100):
                dd = True
            r1.append(rr)
            d1.append(dd)
        rs = deepcopy(r1) 
        rs.sort()
        print('total and mean: ', len(rs), vmean(rs), vmean(d1))
        # print_2f(".25 .5 1  2.5  5  7.5  10  12.5  15  20  25  30  35  40  45  50  75  100 ")
        # print_2f(rs[5], rs[10], rs[20], rs[50], rs[100], rs[150], rs[200], rs[250], rs[300], rs[400], rs[500], rs[600], rs[700], rs[800], rs[900], rs[1000], rs[1500], rs[1999])
        print_vpercent(rs)
    
# test_intial_state('/home/lclan/spinningup/data/', 'Walker2d-v3', 'Walker2d-v3_sac_base', [5e-3, 5e-2, 5e-1])
# test_intial_state('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_sac_base', [0.1, 1.0])
# test_intial_state('/home/lclan/spinningup/data/', 'Humanoid-v3', 'Humanoid-v3_sac_base', [1e-2, 0.1])
# test_intial_state('/home/lclan/spinningup/data/', 'HalfCheetah-v3', 'HalfCheetah-v3_sac_base', [0.1, 1.0, 2.0])
# test_intial_state('/home/lclan/spinningup/data/', 'Hopper-v3', 'Hopper-v3_sac_base', [5e-3, 5e-2, 5e-1])

# test_intial_state('/home/lclan/spinningup/data/', 'Walker2d-v3', 'Walker2d-v3_sac_base', [0.1, 0.25])
# test_intial_state('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_sac_base', [0.25, 0.5])
# test_intial_state('/home/lclan/spinningup/data/', 'HalfCheetah-v3', 'HalfCheetah-v3_sac_base', [0.25, 0.5])
# test_intial_state('/home/lclan/spinningup/data/', 'Hopper-v3', 'Hopper-v3_sac_base', [0.1, 0.25])
# test_intial_state('/home/lclan/spinningup/data/', 'Humanoid-v3', 'Humanoid-v3_sac_base', [0.05, 0.25, 0.5])

# test_intial_state('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_td3_base', [0.1, 0.25, 0.5, 1.0])
# test_intial_state('/home/lclan/spinningup/data/', 'HalfCheetah-v3', 'HalfCheetah-v3_td3_base', [0.1, 0.25, 0.5, 1.0, 2.0])
# test_intial_state('/home/lclan/spinningup/data/', 'Hopper-v3', 'Hopper-v3_td3_base', [0.005, 0.05, 0.1, 0.25, 0.5])
# test_intial_state('/home/lclan/spinningup/data/', 'Walker2d-v3', 'Walker2d-v3_td3_base', [0.005, 0.05, 0.1, 0.25, 0.5])

# test_ppo_intial_state_2('/home/lclan/spinningup/ppo/', 'Ant-v3', 'vanilla_ppo_ant', [0.1, 0.25, 0.5, 1.0])
# test_ppo_intial_state_2('/home/lclan/spinningup/ppo/', 'Walker2d-v3', 'vanilla_ppo_walker', [0.005, 0.05, 0.1, 0.25, 0.5])
# test_ppo_intial_state_2('/home/lclan/spinningup/ppo/', 'Hopper-v3', 'vanilla_ppo_hopper', [0.005, 0.05, 0.1, 0.25, 0.5])
# test_ppo_intial_state_2('/home/lclan/spinningup/ppo/', 'HalfCheetah-v3', 'vanilla_ppo_halfcheetah', [0.1, 0.25, 0.5, 1.0, 2.0])

# test_ppo_intial_state_2('/home/lclan/spinningup/ppo/', 'Ant-v3', 'atla_ppo_ant', [0.1, 0.25, 0.5, 1.0])
# test_ppo_intial_state_2('/home/lclan/spinningup/ppo/', 'Walker2d-v3', 'atla_ppo_walker', [0.005, 0.05, 0.1, 0.25, 0.5])
# test_ppo_intial_state_2('/home/lclan/spinningup/ppo/', 'Hopper-v3', 'atla_ppo_hopper', [0.005, 0.05, 0.1, 0.25, 0.5])
# test_ppo_intial_state_2('/home/lclan/spinningup/ppo/', 'HalfCheetah-v3', 'atla_ppo_halfcheetah', [0.1, 0.25, 0.5, 1.0, 2.0])

test_intial_state('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_sac_rns25', [0.1, 0.25, 0.5, 1.0])
# test_intial_state('/home/lclan/spinningup/data/', 'HalfCheetah-v3', 'HalfCheetah-v3_td3_base', [0.1, 0.25, 0.5, 1.0, 2.0])
# test_intial_state('/home/lclan/spinningup/data/', 'Hopper-v3', 'Hopper-v3_td3_base', [0.005, 0.05, 0.1, 0.25, 0.5])
# test_intial_state('/home/lclan/spinningup/data/', 'Walker2d-v3', 'Walker2d-v3_td3_base', [0.005, 0.05, 0.1, 0.25, 0.5])
