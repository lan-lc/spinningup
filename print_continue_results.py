import gym
from copy import deepcopy
import os
import os.path as osp
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


def print_2f(*args):
    __builtins__.print(*("%.2f" % (a if isinstance(a, float) else a)
                         for a in args), sep='',  end= '')

def print_percent_2f(ls):
    ls = [x*100 for x in ls]
    print_2f(*ls)


def print_rets(rets):
    rets = np.array(rets)
    print("mean, max, min, std", np.mean(rets),np.max(rets) , np.min(rets) , np.std(rets))
    return np.mean(rets)

def get_mean(trajs):
    rets  = deepcopy(trajs[2])
    return sum(rets) / len(rets)

def load_all_same_env_results(fpath, env_name):
    print(fpath)
    file_names = os.listdir(fpath)
    rets = []
    for file_name in file_names:
        if ".pkl" in file_name and get_env_name(file_name) == env_name:
            print(file_name)
            file_name = osp.join(fpath, file_name)
            with open(file_name, 'rb') as f: 
                ret = pickle.load(f)
            rets.append(ret)
    return rets

def is_fail(data):
    if data[3][0]:
        return 1.0
    # if data[3][1] < 100:
    #     return 1.0
    return 0.0
    
def get_fail_rate(result): # result of one agent to one agent's trajs
    ret = 0
    for data in result:
        ret += is_fail(data)
    return ret/len(result)

def get_results_tuple(results):
    stats = {}
    for result in results:
        if (result[0], result[2]) not in stats:
            stats[(result[0], result[2])] = []
        fr = get_fail_rate(result[4])
        print(result[0], result[1], result[2], result[3], fr, len(result[4]))
        if result[1] == result[3]:
            if (result[0], "self") not in stats:
                stats[(result[0], "self")] = []
            stats[(result[0], "self")].append(fr)
        else:
            stats[(result[0], result[2])].append(fr)
    ret = []
    for key in stats.keys():
        np_arr = np.array(stats[key])
        ret.append((key, np.mean(np_arr), np.std(np_arr), np.min(np_arr), np.max(np_arr), len(np_arr)))
        print(key, np.mean(np_arr), np.std(np_arr), np.min(np_arr), np.max(np_arr), len(np_arr) )
    return ret


def print_tuples(tuples, algo_names):
    mean = []
    std = []
    minl = []
    maxl = []
    for algo_name1 in algo_names: # trajs
        mean.append([])
        std.append([])
        minl.append([])
        maxl.append([])
        for tuple in tuples:
            if tuple[0][0] == algo_name1 and tuple[0][1] == 'self':
                mean[-1].append(tuple[1])
                std[-1].append(tuple[2])
                minl[-1].append(tuple[3])
                maxl[-1].append(tuple[4])
                break
            
        for algo_name2 in algo_names: # test_agent
            for tuple in tuples:
                if tuple[0][0] == algo_name1 and tuple[0][1] == algo_name2:
                    mean[-1].append(tuple[1])
                    std[-1].append(tuple[2])
                    minl[-1].append(tuple[3])
                    maxl[-1].append(tuple[4])
                    break
    
    for i in range(len(mean)):
        for j in range(len(mean[i])):
            print_2f(mean[i][j]* 100)
            print(" $\pm$ ", sep='', end='')
            print_2f(std[i][j]* 100)
            print(" & ", sep='', end='')
        print('')
    
    
    # for x in mean:
    #     print_percent_2f(x)
    # for x in std:
    #     print_percent_2f(x)   
    # for x in minl:
    #     print_percent_2f(x)  
    # for x in maxl:
    #     print_percent_2f(x)                   


def print_continue_results(path, env_name, algo_names):
    results = load_all_same_env_results(path, env_name)
    tuples = []
    for result in results:
        tmp = get_results_tuple(result)
        tuples += tmp
    print('tuple:', tuples)
    
    print_tuples(tuples, algo_names)
    

# path  = '/home/lclan/spinningup/data/trajs/test_continue/'
path = '/home/lclan/spinningup/data/tmp/'
algo_names = {}
algo_names['Humanoid-v3'] = ['Humanoid-v3_sac_base', 'Humanoid-v3_td3_base', 'vanilla_ppo_humanoid',  'sgld_ppo_humanoid']
algo_names['Ant-v3'] = ['Ant-v3_sac_base' , 'Ant-v3_td3_base', 'vanilla_ppo_ant', 'atla_ppo_ant']
algo_names['Walker2d-v3'] = ['Walker2d-v3_sac_base', 'Walker2d-v3_td3_base', 'vanilla_ppo_walker', 'atla_ppo_walker']
algo_names['HalfCheetah-v3'] = ['HalfCheetah-v3_sac_base', 'HalfCheetah-v3_td3_base',  'vanilla_ppo_halfcheetah', 'atla_ppo_halfcheetah']
algo_names['Hopper-v3'] = ['Hopper-v3_sac_base', 'Hopper-v3_td3_base', 'vanilla_ppo_hopper',  'atla_ppo_hopper']
env_names =  ['Ant-v3', 'Humanoid-v3', 'Walker2d-v3', 'Hopper-v3', 'HalfCheetah-v3']
parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, required = True)
args =  parser.parse_args()
print_continue_results(path, env_names[args.id], algo_names[env_names[args.id]])


