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
import pickle
def print_2f(*args):
    __builtins__.print(*("%.2f" % a if isinstance(a, float) else a
                         for a in args))

def get_action(o, md, deterministic=True):
    o = torch.as_tensor(o, dtype=torch.float32)
    a = md.act(o)
    return a

def save_state(env):
    return env.sim.get_state()

def restore_state(env, old_state):
    env.reset()
    env.sim.set_state(old_state)
    env.sim.forward()
    return env.get_obs()

# each 
def test_model(env, model, max_ep_len=None, num_episodes=20, interval = 1):
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    total_rewards = []
    trajs = []
    mid_points = []
    while n < num_episodes:
        old_state = save_state(env)
        mid_points.append(old_state)
        if len(mid_points)-1 != ep_len:
            print("bug:", len(mid_points)-1, ep_len)
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

def vmean(v):
    return sum(v) / len(v)

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

def get_models(path, env_name, name):
    env = gym.make(env_name)
    fpath = osp.join(path, name)
    # print(fpath)
    models = []
    while 1:
        fname = osp.join(fpath, name + "_s" + str(len(models)) ,'pyt_save', 'model.pt')
        print(fname)
        if(osp.exists(fname)):
            model = torch.load(fname)
            models.append(model)
        else:
            break
    return models

def gen_good_traj(path, env_name, name, num=1000):
    models = get_models(path, env_name, name)
    env  = gym.make(env_name)
    t = []
    for m in models:
        rr, tt = get_good_trajs(env, m, num)
        t = t + tt
    tp = os.path.join(path, name + "_trajs.pkl")
    print(tp)
    with open(tp, 'wb') as f:
        pickle.dump(t, f)
        
path = '/home/lclan/spinningup/data/'
parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, required = True)
args =  parser.parse_args()
num=1000

if args.id == 0:
    gen_good_traj(path, 'Walker2d-v3', 'Walker2d-v3_sac_base', num)
elif  args.id == 1:
    gen_good_traj(path, 'Ant-v3', 'Ant-v3_sac_base', num)
elif  args.id == 2:
    gen_good_traj(path, 'HalfCheetah-v3', 'HalfCheetah-v3_sac_base', num)
elif  args.id == 3:
    gen_good_traj(path, 'Hopper-v3', 'Hopper-v3_sac_base', num)

