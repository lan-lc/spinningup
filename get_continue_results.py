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
    o = torch.as_tensor(o, dtype=torch.float32)
    return md.act(o)

def print_rets(rets):
    rets = np.array(rets)
    print("mean, max, min, std", np.mean(rets),np.max(rets) , np.min(rets) , np.std(rets))
    return np.mean(rets)

def run_extra_steps(env, ep_len, md, md_name, step_num=50):
    max_ep_len = 1000
    total_r = 0
    o = env.get_obs()
    for i in range(step_num):
        a = get_action(o, md, md_name)
        o, r, d, _ = env.step(a)
        total_r += r
        ep_len += 1
        if d or (ep_len == max_ep_len):
            return (d, total_r)
    return (d, total_r)


def get_all_traj_names_with_same_env(path, trajs_path, name):
    all_trajs_names = []
    fpath = osp.join(path, trajs_path)
    print(fpath)
    file_names = os.listdir(fpath)
    if len(file_names) == 0:
        return []
    env_name = get_env_name(name)
    for file_name in file_names:
        if "trajs.pkl" not in file_name:
            continue
        tmp = get_env_name(file_name)
        if tmp == env_name:
            all_trajs_names.append(file_name)
    return all_trajs_names

def get_mean(trajs):
    rets  = deepcopy(trajs[2])
    return sum(rets) / len(rets)

def get_worst_models_names(path, trajs_path, all_trajs_names, num):
    names = []
    for trajs_name in all_trajs_names: # *4
        trajs_file_name = osp.join(path, trajs_path, trajs_name)
        tmp = []
        with open(trajs_file_name, 'rb') as f: 
            all_trajs = pickle.load(f)    
        for trajs in all_trajs:
            x = (trajs[1], get_mean(trajs))
            print(x)
            if len(tmp) < num:
                tmp.append(x)
            else:
                for j in range(num):
                    if x[1] < tmp[j][1]:
                        tmp[j] = x
                        break
        print(tmp)
        for j in range(len(tmp)):
            names.append(tmp[j][0])
    return names
                
def get_trajs_thr(trajs, top_ratio):
    rets = deepcopy(trajs[2])
    rets.sort()
    return rets[int(len(rets) * (1-top_ratio))]


def sample_midpoint_from_trajs(trajs):
    traj_id = random.randrange(len(trajs[-1]))
    midpoint_id = random.randrange(len(trajs[-1][traj_id]))
    return traj_id, midpoint_id
    

# total interactions will be test_num * step_num * number of agents * number of agents that generate trajs
# 20 * 500 * 10 * 10
def run_extra_steps_on_trajs(model, trajs, test_num, step_num, thr): # 1 model vs trajs of 1 model
    max_len = 999
    ret = []
    env_name = get_env_name(trajs[1])
    env = gym.make(env_name)
    fail_number = 0
    for _ in range(test_num):
        while True:
            traj_id, midpoint_id = sample_midpoint_from_trajs(trajs)
            ep_len = trajs[-1][traj_id][midpoint_id][0]
            if (trajs[2][traj_id] >= (thr-0.1)) and (ep_len + step_num < max_len):
                break
        restore_state(env, trajs[-1][traj_id][midpoint_id][1]) 
        ep_len = trajs[-1][traj_id][midpoint_id][0] 
        tmp = run_extra_steps(env, ep_len, model[2], model[1], step_num)
        if tmp[0]:
            fail_number += 1
        old_ret = sum(trajs[-2][traj_id][ep_len : ep_len + step_num])
        ret.append((traj_id, midpoint_id, old_ret, tmp))
        
    return ret, fail_number/test_num


def is_fail(data):
    if data[3][1] < 500:
        return 1.0
    return 0.0
    
def get_fail_rate(result): # result of one agent to one agent's trajs
    ret = 0
    for data in result:
        ret += is_fail(data)
    return ret/len(result)
    

def print_results(results):
    stats = {}
    for result in results:
        if result[0] not in stats:
            stats[result[0]] = []
        stats[result[0]].append(get_fail_rate(result[4]))
    for key in stats.keys():
        np_arr = np.array(stats[key])
        print(key, np.mean(np_arr), np.std(np_arr))

# def test_all_condinue_old(path, trajs_path, name, test_num, step_num, top_ratio=0.5):
#     models = get_models(path, name) # same algorithm
#     for model in models: 
#         print("model name: ", model[1])
#     all_trajs_names = get_all_traj_names_with_same_env(path, trajs_path, name)
#     print("all traj file name: ", all_trajs_names)
#     worst_model_names = get_worst_models_names(path, trajs_path, all_trajs_names, 2)
#     print("worst model names: ", worst_model_names)
#     ret = []
#     for trajs_name in all_trajs_names: # *4
#         trajs_file_name = osp.join(path, trajs_path, trajs_name)
#         print("start runing on: ", trajs_file_name)
#         with open(trajs_file_name, 'rb') as f: 
#             all_trajs = pickle.load(f)
#         for trajs in all_trajs: # * 10
#             if trajs[1] in worst_model_names:
#                 continue
#             thr = get_trajs_thr(trajs, top_ratio)
#             print("running with ", trajs[0], trajs[1], thr) 
#             # print(trajs[0], trajs[1], type(trajs[2][0]), len(trajs[3][0]),  len(trajs[4][0])) # float , 1000, 100
#             for model in models: # * 12 or 11
#                 if model[1] == trajs[1] or model[1] in worst_model_names:
#                     continue
#                 tmp = run_extra_steps_on_trajs(model, trajs, test_num, step_num, thr)
#                 ret.append((trajs[0], trajs[1], model[0], model[1], tmp))    
#             del trajs
#     result_name = name + "_s" + str(step_num) + "_tr" + str(int(top_ratio*100)) + ".pkl"
#     result_path = osp.join(path, result_name)
#     with open(result_path , 'wb') as f:
#         pickle.dump(ret, f)
#     print_results(ret)
    
    
def get_trajs_name_by_algo(path, trajs_path, algo_name):
    all_trajs_names = get_all_traj_names_with_same_env(path, trajs_path, algo_name)
    for trajs_name in all_trajs_names:
        if algo_name in trajs_name:
            return trajs_name

def get_all_models_with_same_env(path, algo_name, all_algo_names):
    model_names = []
    models = {}
    env_name = get_env_name(algo_name)
    for x in all_algo_names:
        tmp_env_name = get_env_name(x)
        if tmp_env_name == env_name:
            model_names.append(x)
    print(model_names)
    for x in model_names:
        models[x] = get_models(path, x) # same algorithm
    return models
    

def get_model(models, model_name):
    print(model_name)
    for model in models:
        if model[1] == model_name:
            return model

def run_extra_steps_on_easy_states(trajs, self_run, model, test_num, step_num):
    ret = []
    env_name = get_env_name(trajs[1])
    env = gym.make(env_name)
    fail_number = 0
    for i in range(len(self_run)):
        if self_run[i][3][0]: #done
            continue
        traj_id, midpoint_id = self_run[i][0], self_run[i][1]
        restore_state(env, trajs[-1][traj_id][midpoint_id][1])
        ep_len = trajs[-1][traj_id][midpoint_id][0] 
        tmp = run_extra_steps(env, ep_len, model[2], model[1], step_num)
        old_ret = sum(trajs[-2][traj_id][ep_len : ep_len + step_num])
        ret.append((traj_id, midpoint_id, old_ret, tmp))
        if tmp[0]:
            fail_number += 1
        if len(ret) == test_num:
            print(model[1], "fail rate", fail_number/len(ret))
            return ret, fail_number/len(ret)
    print(model[1], "fail rate", fail_number/len(ret))
    return ret, fail_number/len(ret)
        


def test_all_condinue(path, trajs_path, algo_name, all_algo_names, test_num, step_num, top_ratio=0.5):
    all_trajs_names = get_all_traj_names_with_same_env(path, trajs_path, algo_name)
    print("all traj file name: ", all_trajs_names)
    worst_model_names = get_worst_models_names(path, trajs_path, all_trajs_names, 2)
    # worst_model_names = ['Ant-v3_td3_base_s112', 'Ant-v3_td3_base_s115', 'vanilla_ppo_ant_4.pt', 'vanilla_ppo_ant_6.pt', 'atla_ppo_ant_1.pt', 'atla_ppo_ant_0.pt', 'Ant-v3_sac_base_s1209', 'Ant-v3_sac_base_s1204']
    print("worst model names: ", worst_model_names)
    trajs_name = get_trajs_name_by_algo(path, trajs_path, algo_name)
    print("all traj file name: ", trajs_name)
    models = get_all_models_with_same_env(path, algo_name, all_algo_names)
    ret = []
    trajs_file_name = osp.join(path, trajs_path, trajs_name)
    print("start loading ", trajs_file_name)
    with open(trajs_file_name, 'rb') as f: 
        all_trajs = pickle.load(f)
    ret = []
    self_frs = []
    for trajs in all_trajs:
        if trajs[1] in worst_model_names:
            continue
        thr = get_trajs_thr(trajs, top_ratio)
        print("running on trajs of ", trajs[0], trajs[1], thr)
        model = get_model(models[algo_name], trajs[1])
        self_run, fr = run_extra_steps_on_trajs(model, trajs, test_num * 8 , step_num, thr)
        print(model[1], "self play fail rate: ", fr)
        self_frs.append(fr)
        if fr > 0.2:
            continue
        ret.append((trajs[0], trajs[1], model[0], model[1], self_run))
        for k in models.keys():
            frs = []
            for model in models[k]:
                print("running with model ", model[0], model[1])
                tmp, fr = run_extra_steps_on_easy_states(trajs, self_run, model, test_num, step_num)
                ret.append((trajs[0], trajs[1], model[0], model[1], tmp)) 
                frs.append(fr)
            print_rets(frs)
    result_name = algo_name + "_s" + str(step_num) + "_tr" + str(int(top_ratio*100)) + "_tn" + str(test_num) + ".pkl"
    result_path = osp.join(path, result_name)
    print('self rate ')
    print_rets(self_frs)
    with open(result_path , 'wb') as f:
        pickle.dump(ret, f)
    print_results(ret)


path  = '/home/lclan/spinningup/data/'
trajs_path = 'trajs'
step_num = 500
test_num = 200
parser = argparse.ArgumentParser()
parser.add_argument('--name_id', type=int, required = True)
names = ['Ant-v3_sac_base' , 'Ant-v3_td3_base', 'vanilla_ppo_ant', 'atla_ppo_ant', 
        'Humanoid-v3_sac_base', 'Humanoid-v3_td3_base', 'vanilla_ppo_humanoid',  'sgld_ppo_humanoid',
        'Walker2d-v3_sac_base', 'Walker2d-v3_td3_base', 'vanilla_ppo_walker', 'atla_ppo_walker',
        'HalfCheetah-v3_sac_base', 'HalfCheetah-v3_td3_base',  'vanilla_ppo_halfcheetah', 'atla_ppo_halfcheetah',  
        'Hopper-v3_sac_base', 'Hopper-v3_td3_base', 'vanilla_ppo_hopper',  'atla_ppo_hopper' ]
args =  parser.parse_args()
test_all_condinue(path, trajs_path, names[args.name_id], names, test_num, step_num)