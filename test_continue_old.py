from ast import mod
from turtle import mode
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
def print_2f(*args):
    __builtins__.print(*("%.2f" % a if isinstance(a, float) else a
                         for a in args))

def get_action(o, md, deterministic=True):
    o = torch.as_tensor(o, dtype=torch.float32)
    a = md.act(o)
    return a

def test_model(env, model, max_ep_len=None, num_episodes=20, interval = 1):
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    total_rewards = []
    trajs = []
    mid_points = {}
    while n < num_episodes:
        if ep_len % interval == 0:
            old_state = save_state(env)
            mid_points[ep_len] = old_state
        a = get_action(o, model)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        if d or (ep_len == max_ep_len):
            total_rewards.append(ep_ret)
            trajs.append(mid_points)
            mid_points = {}
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1
    return total_rewards, trajs

def get_good_trajs(env, model, num, max_ep_len=1000):
    total_rewards = []
    trajs = []
    old_r = []
    while len(total_rewards) < num:
        r, t = test_model(env, model, max_ep_len, num - len(total_rewards))
        old_r = old_r + r
        thr = vmean(old_r)
        for i in range(len(r)):
            if thr <= r[i] and len(t[i])>=500:
                total_rewards.append(r[i])
                trajs.append(t[i])
    return total_rewards, trajs


def get_models(path, env_name, name):
    env = gym.make(env_name)
    fpath = osp.join(path, name)
    # print(fpath)
    models = []
    while 1:
        if 'ppo' in name:
            fname = osp.join(fpath, name + "_" + str(len(models)) + '.pt')
        else:
            fname = osp.join(fpath, name + "_s" + str(len(models)) ,'pyt_save', 'model.pt')
        # print(fname)
        if(osp.exists(fname)):
            if 'ppo' in fname:
                env = gym.make(env_name)
                obs_dim = env.observation_space.shape[0]
                action_dim = env.action_space.shape[0]
                model = PPO_Actor(obs_dim, action_dim, (64, 64), nn.Tanh)
                model.load(fname)
                models.append(model)
            else:
                model = torch.load(fname)
                models.append(model)
        else:
            break
    # print(len(models))
    test_val = []
    for model in models:
        x, _ = test_model(env, model, num_episodes=20)
        # print(x, stats.trim_mean(x, 0.1))
        test_val.append(stats.trim_mean(x, 0.1))
    print_2f(test_val)
    sorted_ids = np.argsort(test_val)
    model1 = models[sorted_ids[-1]]
    model2 = models[sorted_ids[-2]]
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
    env.reset()
    env.sim.set_state(old_state)
    env.sim.forward()
    return env.get_obs()

def vmean(v):
    return sum(v) / len(v)

def print_vpercent(v, sp = [1, 2.5, 5, 10, 15, 20, 30, 40, 50, 60, 70, 75, 80, 90, 99], print_sp = True):
    l = len(v)
    if print_sp:
        print_2f(*sp)
    vs = [v[int(l*x/100)] for x in sp]
    print_2f(*vs)
    return vs
    

# using m1 to generate good traj and test on m2
def test_continus(env, m1, m2, num=1000):
    rs, trajs = get_good_trajs(env, m1, num)
    print_2f("good trajs mean: ", vmean(rs))
    r1, d1, r2, d2 = [], [], [], []
    for t in trajs:
        old_state = t[490]
        o = restore_state(env, old_state)
        dd, rr = run_extra_steps(env, o, 490, m1, 1000, 500)
        if(rr < 500):
            dd = True
        r1.append(rr)
        d1.append(dd)
        old_state = t[499]
        o = restore_state(env, old_state)
        dd, rr = run_extra_steps(env, o, 490, m2, 1000, 500)
        if(rr < 500):
            dd = True
        r2.append(rr)
        d2.append(dd)
    # print("r1 d1 r2 d2", vmean(r1), vmean(d1), vmean(r2), vmean(d2))   
    # print("r1")
    # r1.sort() 
    # print_vpercent(r1)
    # print("r2")
    # r2.sort()
    # print_vpercent(r2)
    return r1, d1, r2, d2
    
    
# def test(path, env_name, name):
#     model1, model2 = get_models(path, env_name, name)
#     env = gym.make(env_name)
#     print("generate traj with m1")
#     test_continus(env, model1, model2)
#     print("generate traj with m2")
#     test_continus(env, model2, model1)
   
# test('/home/lclan/spinningup/data/', 'Walker2d-v3', 'Walker2d-v3_sac_base')
# test('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_sac_base')
# test('/home/lclan/spinningup/data/', 'Humanoid-v3', 'Humanoid-v3_sac_base')
# test('/home/lclan/spinningup/data/', 'HalfCheetah-v3', 'HalfCheetah-v3_sac_base')
# test('/home/lclan/spinningup/data/', 'Hopper-v3', 'Hopper-v3_sac_base')


    
  
def print_rd12(r1, d1, r2, d2):
    print("r1 d1 r2 d2", vmean(r1), vmean(d1), vmean(r2), vmean(d2))   
    print("r1")
    r1.sort() 
    print_vpercent(r1)
    print("r2")
    r2.sort()
    print_vpercent(r2)
    
def print_rd(r, d):
    r.sort() 
    print("r d: ", vmean(r), vmean(d))
    print_vpercent(r, print_sp=False)
  
def test_same_algorithm(env, models, num=1000):
    r1, r2, d1, d2 = [], [], [], []
    for i in range(len(models)):
        for j in range(len(models)):
            if i != j:
                tmp = test_continus(env, models[i], models[j], num)
                r1 = r1 + tmp[0]
                d1 = d1 + tmp[1]
                r2 = r2 + tmp[2]
                d2 = d2 + tmp[3]
    print_rd12(r1, d1, r2, d2)

def test_two_algorithm(env, algo1_models, algo2_models, num=1000):
    r1, r2, d1, d2 = [], [], [], []       
    for i in range(len(algo1_models)):
        for j in range(len(algo2_models)):
            tmp = test_continus(env, algo1_models[i], algo2_models[j], num)
            r1 = r1 + tmp[0]
            d1 = d1 + tmp[1]
            r2 = r2 + tmp[2]
            d2 = d2 + tmp[3]
    print_rd12(r1, d1, r2, d2)

def test_diff_algo(path, env_name, name1, name2, num=1000):
    model11, model12 = get_models(path, env_name, name1)
    model21, model22 = get_models(path, env_name, name2)
    env = gym.make(env_name)
    algo1_models = [model11, model12]
    algo2_models = [model21, model22]
    print("self_test of ", name1)
    test_same_algorithm(env, algo1_models, num)
    print("self_test of ", name2)
    test_same_algorithm(env, algo2_models, num)
    print("test ", name2, " on ", name1, "'s trajs")
    test_two_algorithm(env, algo1_models, algo2_models, num)
    print("test ", name1, " on ", name2, "'s trajs")
    test_two_algorithm(env, algo2_models, algo1_models, num)


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
    
def test_ppo(path, env_name, name, ppo_name, num=1000):
    model1, model2 = get_models(path, env_name, name)
    env = gym.make(env_name)
    algo_models = [model1, model2]
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = PPO_Actor(obs_dim, action_dim, (64, 64), nn.Tanh)
    model.load(ppo_name)
    ppo_models = [model]
    print("generate traj with ", name)
    test_two_algorithm(env, algo_models, ppo_models, num)
    
    # print("generate traj with m1")
    # test_continus(env, model1, model2)
    # print("generate traj with m2")
    # test_continus(env, model2, model1)

def test_models_with_trajs(env, trajs, models):
    r = []
    d = []
    d = []
    step_nums = [400, 411, 434, 471]
    for t in trajs:
        for m in models:
            for step_num in step_nums:
                old_state = t[step_num-1]
                o = restore_state(env, old_state)
                dd, rr = run_extra_steps(env, o, step_num, m, 1000, 500)
                if(rr < 500):
                    dd = True
                r.append(rr)
                d.append(dd)
    return r, d
    
    
def test_continue(path, env_name, traj_name , names, num=250):
    m1, m2 = get_models(path, env_name, traj_name)
    env = gym.make(env_name)
    rs1, trajs1 = get_good_trajs(env, m1, num)
    rs2, trajs2 = get_good_trajs(env, m2, num)
    d = []
    print('avg of m1 m2 trajs: ', vmean(rs1), vmean(rs2))
    print('same algorithm', traj_name)
    r1 = []
    d1 = []
    rr, dd = test_models_with_trajs(env, trajs1, [m1])
    r1 = r1 + rr
    d1 = d1 + dd
    rr, dd = test_models_with_trajs(env, trajs2, [m2])
    r1 = r1 + rr
    d1 = d1 + dd
    r2 = []
    d2 = []
    rr, dd = test_models_with_trajs(env, trajs1, [m2])
    r2 = r2 + rr
    d2 = d2 + dd
    rr, dd = test_models_with_trajs(env, trajs2, [m1])
    r2 = r2 + rr
    d2 = d2 + dd
    print_rd12(r1, d1, r2, d2)
    d.append(vmean(d1))
    d.append(vmean(d2))
    trajs = trajs1 + trajs2
    for name in names:
        print("test ", name, " on ", traj_name, "'s trajs")
        m1, m2 = get_models(path, env_name, name)
        r2 , d2 = test_models_with_trajs(env, trajs, [m1,m2])
        print_rd(r2, d2)
        d.append(vmean(d2))
    print(d)
        

def test_full_continue(path, env_name, names, num=250):
    for i in range(len(names)):
        tmp = []
        for j in range(len(names)):
            if i!=j:
                tmp.append(names[j])
        test_continue(path, env_name, names[i], tmp, num)
        

path = '/home/lclan/spinningup/data/'

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, required = True)
args =  parser.parse_args()

if args.id == 0:
    test_full_continue(path, 'Walker2d-v3', ['Walker2d-v3_sac_base', 'Walker2d-v3_td3_base', 'vanilla_ppo_walker', 'atla_ppo_walker'])
elif  args.id == 1:
    test_full_continue(path, 'Ant-v3', ['Ant-v3_sac_base', 'Ant-v3_td3_base', 'vanilla_ppo_ant', 'atla_ppo_ant'])
elif  args.id == 2:
    test_full_continue(path, 'HalfCheetah-v3', ['HalfCheetah-v3_sac_base', 'HalfCheetah-v3_td3_base', 'vanilla_ppo_halfcheetah', 'atla_ppo_halfcheetah'])
elif  args.id == 3:
    test_full_continue(path, 'Hopper-v3', ['Hopper-v3_sac_base', 'Hopper-v3_td3_base', 'vanilla_ppo_hopper', 'atla_ppo_hopper'])



# test_diff_algo('/home/lclan/spinningup/data/', 'Walker2d-v3', 'Walker2d-v3_sac_base', 'Walker2d-v3_td3_base')
# test_diff_algo('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_sac_base', 'Ant-v3_td3_base')
# test_diff_algo('/home/lclan/spinningup/data/', 'HalfCheetah-v3', 'HalfCheetah-v3_sac_base', 'HalfCheetah-v3_td3_base')
# test_diff_algo('/home/lclan/spinningup/data/', 'Hopper-v3', 'Hopper-v3_sac_base', 'Hopper-v3_td3_base')
# test_diff_algo('/home/lclan/spinningup/data/', 'Humanoid-v3', 'Humanoid-v3_sac_base', 'Humanoid-v3_td3_base')

# test_ppo('/home/lclan/spinningup/data/', 'Walker2d-v3', 'Walker2d-v3_sac_base', './ppo-walker.pt')
# test_ppo('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_sac_base', './ppo-ant.pt')
# test_ppo('/home/lclan/spinningup/data/', 'HalfCheetah-v3', 'HalfCheetah-v3_sac_base', './ppo-halfcheetah.pt')
# test_ppo('/home/lclan/spinningup/data/', 'Hopper-v3', 'Hopper-v3_sac_base', './ppo-hopper.pt')

# test_ppo('/home/lclan/spinningup/data/', 'Walker2d-v3', 'Walker2d-v3_td3_base', './ppo-walker.pt')
# test_ppo('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_td3_base', './ppo-ant.pt')
# test_ppo('/home/lclan/spinningup/data/', 'HalfCheetah-v3', 'HalfCheetah-v3_td3_base', './ppo-halfcheetah.pt')
# test_ppo('/home/lclan/spinningup/data/', 'Hopper-v3', 'Hopper-v3_td3_base', './ppo-hopper.pt')