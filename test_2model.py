from ast import mod
import gym
from copy import deepcopy
import os
import os.path as osp
import torch
from scipy import stats
from statistics import mean 
import numpy as np

def print_2f(*args):
    __builtins__.print(*("%.2f" % a if isinstance(a, float) else a
                         for a in args))


def get_action(o, md):
    with torch.no_grad():
        o = torch.as_tensor(o, dtype=torch.float32)
        a = md.act(o)
    return a

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


def test_model(env, model, max_ep_len=None, num_episodes=20, interval = 200):
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



def get_good_trajs(env, model, max_ep_len=1000, num_episodes=20, interval = 200):
    total_rewards = []
    trajs = []
    while len(total_rewards) < num_episodes:
        r, t = test_model(env, model, max_ep_len, num_episodes - len(total_rewards), interval)
        thr = sum(r) / len(r)
        for i in range(len(r)):
            if r[i] > thr-1:
                total_rewards.append(r[i])
                trajs.append(t[i])
    return total_rewards, trajs

def run_policy(env, md, md2, max_ep_len=1000, num_episodes=1):
    log = {}
    log2 = {} 
    for i in range((int)(1000/50)-1):
        log[(i+1)*50] = []
        log2[(i+1)*50] = []
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    old_o = o
    old_r = ep_ret
    old_env = gym.make(env.name)
    old_env.reset()
    old_state = deepcopy(env.sim.get_state())
    old_env.sim.set_state(old_state)
    old_env.sim.forward()
    old_ep_len = ep_len
    v_ret = []
    ds = []
    while n < num_episodes:
        a = get_action(o, md)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        if ep_len in log:
            log[ep_len].append(ep_ret - old_r)
            dd, rr = run_extra_steps(old_env, old_o, old_ep_len, md2, max_ep_len, 50)
            log2[ep_len].append(rr)
            ds.append(dd)
            old_o = o
            old_r = ep_ret
            old_state = deepcopy(env.sim.get_state())
            old_env.sim.set_state(old_state)
            old_env.sim.forward()
            old_ep_len = ep_len
            
        if d or (ep_len == max_ep_len):
            v_ret.append(ep_ret)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            old_o = o
            old_r = ep_ret
            # old_env = deepcopy(env)
            old_env.reset()
            old_state = deepcopy(env.sim.get_state())
            old_env.sim.set_state(old_state)
            old_env.sim.forward()
            old_ep_len = ep_len
            n += 1
    
    v = []
    v2 = []
    for key in log.keys():
        # print(key, sum(log[key]) / len(log[key]), sum(log2[key]) / len(log2[key]), len(log[key]))
        v.append(sum(log[key]) / len(log[key]))
        v2.append(sum(log2[key]) / len(log2[key]))
    print_2f(sum(v) / len(v), sum(v2) / len(v2), sum(v_ret) / len(v_ret), len(ds), sum(ds) / len(ds)) 



def test_intial_state(path, env_name, name, noise_scales):
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
    env = gym.make(env_name)
    for model in models:
        x, _ = test_model(env, model, num_episodes=5)
        # print(x, stats.trim_mean(x, 0.1))
        test_val.append(stats.trim_mean(x, 0.1))
    print_2f(test_val)
    sorted_ids = np.argsort(test_val)
    model1 = models[sorted_ids[-1]]
    model2 = models[sorted_ids[-2]]
    for ns in noise_scales:
        print_2f("ns: ",  ns)
        env = gym.make(env_name, reset_noise_scale=ns)
        r1 = []
        d1 = []
        r2 = []
        d2 = []
        for _ in range(100):
            o = env.reset()
            state = deepcopy(env.sim.get_state())
            dd, rr = run_extra_steps(env, o, 0, model1, 1000, 200)
            r1.append(rr)
            d1.append(dd)
            env.reset()
            env.sim.set_state(state)
            env.sim.forward()
            dd, rr = run_extra_steps(env, o, 0, model2, 1000, 200)
            r2.append(rr)
            d2.append(dd)
        
        print_2f("average and min m1 m2: ", sum(r1) / len(r1), sum(r2) / len(r2), min(r1), min(r2))
        print_2f("done m1, m2", sum(d1) / len(d1), sum(d2) / len(d2) )     
        total = len(d1)
        cnt = [0,0,0,0]
        for i in range(total):
            c = 0
            if d1[i]:
                c+=1
            if d2[i]:
                c+=2
            cnt[c] += 100/(total)
        print_2f("00, 01, 10, 11 ", cnt, (cnt[1]+cnt[3])*(cnt[2]+cnt[3])/100 )
              
    

def testing(path, env_name, name, noise_scales):
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
    env = gym.make(env_name)
    for model in models:
        x, _ = test_model(env, model, num_episodes=5)
        # print(x, stats.trim_mean(x, 0.1))
        test_val.append(stats.trim_mean(x, 0.1))
    print_2f(test_val)
    sorted_ids = np.argsort(test_val)
    model1 = models[sorted_ids[-1]]
    model2 = models[sorted_ids[-2]]
    for ns in noise_scales:
        print_2f("ns: ",  ns)
        env = gym.make(env_name, reset_noise_scale=ns)
        old_env = gym.make(env_name, reset_noise_scale=ns)
        rs1, ts1 = get_good_trajs(env, model1, 1000, 100, 200)
        t1m1_r = []
        t1m1_d = []
        t1m2_r = []
        t1m2_d = []
        for t in ts1:
            for mid_point in t:
                if mid_point[0] != 0 and  mid_point[0] != 1000:
                    old_env.reset()
                    old_env.sim.set_state(mid_point[1])
                    old_env.sim.forward()
                    dd, rr = run_extra_steps(old_env, mid_point[2], mid_point[0], model1, 1000, 200)
                    t1m1_r.append(rr)
                    t1m1_d.append(dd)
                    old_env.reset()
                    old_env.sim.set_state(mid_point[1])
                    old_env.sim.forward()
                    dd, rr = run_extra_steps(old_env, mid_point[2], mid_point[0], model2, 1000, 200)
                    t1m2_r.append(rr)
                    t1m2_d.append(dd)
        print_2f("t1 average: ", sum(rs1)/len(rs1))
        print_2f("t1m1 v d: ", sum(t1m1_r)/len(t1m1_r), sum(t1m1_d)/len(t1m1_d))
        print_2f("t1m2 v d: ", sum(t1m2_r)/len(t1m2_r), sum(t1m2_d)/len(t1m2_d))
        
        
        rs2, ts2 = get_good_trajs(env, model2, 1000, 20, 200)
        t2m1_r = []
        t2m1_d = []
        t2m2_r = []
        t2m2_d = []
        for t in ts2:
            for mid_point in t:
                if mid_point[0] != 0 and  mid_point[0] != 1000:
                    old_env.reset()
                    old_env.sim.set_state(mid_point[1])
                    old_env.sim.forward()
                    dd, rr = run_extra_steps(old_env, mid_point[2], mid_point[0], model1, 1000, 200)
                    t2m1_r.append(rr)
                    t2m1_d.append(dd)
                    old_env.reset()
                    old_env.sim.set_state(mid_point[1])
                    old_env.sim.forward()
                    dd, rr = run_extra_steps(old_env, mid_point[2], mid_point[0], model2, 1000, 200)
                    t2m2_r.append(rr)
                    t2m2_d.append(dd)

        print_2f("t2 average: ", sum(rs2)/len(rs2))
        print_2f("t2m1 v d: ", sum(t2m1_r)/len(t2m1_r), sum(t2m1_d)/len(t2m1_d))
        print_2f("t2m2 v d: ", sum(t2m2_r)/len(t2m2_r), sum(t2m2_d)/len(t2m2_d))




        

# testing('/home/lclan/spinningup/data/', 'Walker2d-v3', 'Walker2d-v3_sac_base', [5e-3])
# testing('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_sac_base', [0.1])
# testing('/home/lclan/spinningup/data/', 'Humanoid-v3', 'Humanoid-v3_sac_base', [1e-2])
# testing('/home/lclan/spinningup/data/', 'HalfCheetah-v3', 'HalfCheetah-v3_sac_base', [0.1])
# testing('/home/lclan/spinningup/data/', 'Hopper-v3', 'Hopper-v3_sac_base', [5e-3])


# test_intial_state('/home/lclan/spinningup/data/', 'Walker2d-v3', 'Walker2d-v3_sac_base', [5e-3, 5e-2, 5e-1])
# test_intial_state('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_sac_base', [0.1, 1.0])
# test_intial_state('/home/lclan/spinningup/data/', 'Humanoid-v3', 'Humanoid-v3_sac_base', [1e-2, 0.1])
# test_intial_state('/home/lclan/spinningup/data/', 'HalfCheetah-v3', 'HalfCheetah-v3_sac_base', [0.1, 1.0, 2.0])
# test_intial_state('/home/lclan/spinningup/data/', 'Hopper-v3', 'Hopper-v3_sac_base', [5e-3, 5e-2, 5e-1])

testing('/home/lclan/spinningup/data/', 'Walker2d-v3', 'Walker2d-v3_ppo_base', [5e-3])
testing('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_ppo_base', [0.1])
# testing('/home/lclan/spinningup/data/', 'Humanoid-v3', 'Humanoid-v3_ppo_base', [1e-2])
# testing('/home/lclan/spinningup/data/', 'HalfCheetah-v3', 'HalfCheetah-v3_ppo_base', [0.1])
# testing('/home/lclan/spinningup/data/', 'Hopper-v3', 'Hopper-v3_ppo_base', [5e-3])


test_intial_state('/home/lclan/spinningup/data/', 'Walker2d-v3', 'Walker2d-v3_ppo_base', [5e-3, 5e-2, 5e-1])
test_intial_state('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_ppo_base', [0.1, 1.0])
# test_intial_state('/home/lclan/spinningup/data/', 'Humanoid-v3', 'Humanoid-v3_ppo_base', [1e-2, 0.1])
# test_intial_state('/home/lclan/spinningup/data/', 'HalfCheetah-v3', 'HalfCheetah-v3_ppo_base', [0.1, 1.0, 2.0])
# test_intial_state('/home/lclan/spinningup/data/', 'Hopper-v3', 'Hopper-v3_ppo_base', [5e-3, 5e-2, 5e-1])




# fpath = '/home/lclan/spinningup/data/Walker2d-v3_sac_base/Walker2d-v3_sac_base_s0'
# fname = osp.join(fpath, 'pyt_save', 'model.pt')
# model = torch.load(fname)
# fpath2 = '/home/lclan/spinningup/data/Walker2d-v3_sac_base/Walker2d-v3_sac_base_s1'
# fname2 = osp.join(fpath2, 'pyt_save', 'model.pt')
# model2 = torch.load(fname2)


# env_name = 'Walker2d-v3'

# rns = 5e-3
# env = gym.make(env_name, reset_noise_scale=rns)
# print(rns)
# run_policy(env, model, model2, 1000 , 20)
# run_policy(env, model2, model, 1000 , 20)
# run_policy(env, model, model, 1000 , 20)
# run_policy(env, model2, model2, 1000 , 20)

# rns = 5e-2
# env = gym.make(env_name, reset_noise_scale=rns)
# print(rns)
# run_policy(env, model, model2, 1000 , 20)
# run_policy(env, model2, model, 1000 , 20)
# run_policy(env, model, model, 1000 , 20)
# run_policy(env, model2, model2, 1000 , 20)

# rns = 5e-1
# env = gym.make(env_name, reset_noise_scale=rns)
# print(rns)
# run_policy(env, model, model2, 1000 , 20)
# run_policy(env, model2, model, 1000 , 20)
# run_policy(env, model, model, 1000 , 20)
# run_policy(env, model2, model2, 1000 , 20)

# env_name = 'Ant-v3'
# env = gym.make(env_name)
# run_policy(env, model, model2, 1000 , 20)
# run_policy(env, model2, model, 1000 , 20)
# run_policy(env, model, model, 1000 , 20)
# run_policy(env, model2, model2, 1000 , 20)


# fpath = '/home/lclan/spinningup/data/Ant-v3_sac_base/Ant-v3_sac_base_s1'
# fname = osp.join(fpath, 'pyt_save', 'model.pt')
# model = torch.load(fname)
# fpath2 = '/home/lclan/spinningup/data/Ant-v3_sac_base/Ant-v3_sac_base_s2'
# fname2 = osp.join(fpath2, 'pyt_save', 'model.pt')
# model2 = torch.load(fname2)
# env_name = 'Ant-v3'

# rns = 0.1
# env = gym.make(env_name, reset_noise_scale=rns)
# print(rns)
# run_policy(env, model, model2, 1000 , 20)
# run_policy(env, model2, model, 1000 , 20)
# run_policy(env, model, model, 1000 , 20)
# run_policy(env, model2, model2, 1000 , 20)

# rns = 1.0
# env = gym.make(env_name, reset_noise_scale=rns)
# print(rns)
# run_policy(env, model, model2, 1000 , 20)
# run_policy(env, model2, model, 1000 , 20)
# run_policy(env, model, model, 1000 , 20)
# run_policy(env, model2, model2, 1000 , 20)


