from ast import mod
import gym
from copy import deepcopy
import os
import os.path as osp
import torch
from scipy import stats
from statistics import mean 
import numpy as np
import spinup.algos.pytorch.sac.core as core
from torch.optim import Adam
import itertools
import random

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


# print .2
def print_2f(*args):
    __builtins__.print(*("%.2f" % a if isinstance(a, float) else a
                         for a in args))


# get best action
def get_action(o, md):
    with torch.no_grad():
        o = torch.as_tensor(o, dtype=torch.float32)
        a = md.act(o)
    return a


    

# let an agent start playing step_num of steps from a given env with obs = o
def run_extra_steps(env, o, ep_len, md, max_ep_len, step_num = 50):
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

# testing model to select the best model to 
def test_model(env, model, max_ep_len=None, num_episodes=20, interval = 200):
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    total_rewards = []
    trajs = []
    mid_points = []
    while n < num_episodes:
        old_state = deepcopy(env.sim.get_state())
        mid_points.append((ep_len, old_state, o))
        a = get_action(o, model)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        if d or (ep_len == max_ep_len):
            total_rewards.append(ep_ret)
            trajs.append(mid_points)
            mid_points = [] # clear mid point
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1
    return total_rewards, trajs

# get_value by using q(s, pi(s))
def get_value(o, md):
    if not torch.is_tensor(o):
        o = torch.as_tensor(o, dtype=torch.float32)
    with torch.no_grad():
        a, _ = md.pi(o, deterministic=True, with_logprob=False)
        q1 = md.q1(o, a)
        q2 = md.q2(o, a)
        q = torch.min(q1, q2)
    return q


# get_grad of pos and vel
def get_grad(env, md, eps):
    qpos = deepcopy(env.sim.data.qpos)
    qvel = deepcopy(env.sim.data.qvel)
    o = env.get_obs()
    old_q = get_value(o, md).numpy()
    grad_pos = np.zeros_like(qpos)
    for i in range(len(qpos)):
        qpos[i]+=eps
        env.set_state(qpos, qvel)
        o = env.get_obs()
        q = get_value(o, md).numpy()
        grad_pos[i] = (q - old_q) / eps        
        qpos[i]-=eps
        
    grad_vel = np.zeros_like(qvel)
    for i in range(len(qvel)):
        qvel[i]+=eps
        env.set_state(qpos, qvel)
        o = env.get_obs()
        q = get_value(o, md).numpy()
        grad_vel[i] = (q - old_q) / eps        
        qvel[i]-=eps
    return grad_pos, grad_vel




def save_state(env):
    qpos = deepcopy(env.sim.data.qpos)
    qvel = deepcopy(env.sim.data.qvel)
    return (qpos ,qvel)

def restore_state(env, old_state):
    env.reset()
    env.set_state(old_state[0], old_state[1])
    new_obs = env.get_obs()
    return new_obs

def get_state_dis(s1, s2):
    l = len(s1[0]) + len(s1[1])
    l1_mean_dis = (sum(np.abs(s1[0]-s2[0])) + sum(np.abs(s1[1]-s2[1]))) / l
    l2_mean_dis = np.sqrt((sum((s1[0]-s2[0])**2) + sum((s1[1]-s2[1])**2))/l)
    return l1_mean_dis, l2_mean_dis

def get_init_state(env):
    env.sim.reset()
    init_qpos = env.sim.data.qpos.ravel().copy()
    init_qvel = env.sim.data.qvel.ravel().copy()
    return (init_qpos, init_qvel)

def get_mean(v):
    return sum(v)/len(v)

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
    env = gym.make(env_name)
    for model in models:
        x, _ = test_model(env, model, num_episodes=5)
        # print(x, stats.trim_mean(x, 0.1))
        test_val.append(stats.trim_mean(x, 0.1))
    sorted_ids = np.argsort(test_val)
    model1 = models[sorted_ids[-1]]
    return model1


# change the current state to adv state
def change_env_to_adv(env, md, init_state, step_size, eps, use_sign=False):
    qpos = deepcopy(env.sim.data.qpos)
    qvel = deepcopy(env.sim.data.qvel)
    grad_pos, grad_vel = get_grad(env, md, eps)
    
    if use_sign:
        grad_pos = np.sign(grad_pos)
        grad_vel = np.sign(grad_vel)
    else:
        grad_pos = np.clip(grad_pos, -1, 1)
        grad_vel = np.clip(grad_vel, -1, 1)
    
    adv_qpos = qpos - step_size*grad_pos
    adv_qvel = qvel - step_size*grad_vel
    init_qpos = init_state[0]
    init_qvel = init_state[1]
    rns = 0.1
    adv_qpos = np.clip(adv_qpos, init_qpos - rns,  init_qpos + rns)
    adv_qvel = np.clip(adv_qvel, init_qvel - 3.*rns, init_qvel + 3.*rns)
    env.set_state(adv_qpos, adv_qvel)
    new_obs = env.get_obs()
    return new_obs


def test_model_adv(env, md, adv_step_num, adv_step_size, eps=None, sign=True):
    if eps == None:
        eps = adv_step_size
    test_num = 30
    test_l = 1000
    init_state = get_init_state(env)
    l2_old, l2_init = [], []
    orig_rs, adv_rs, orig_qs, adv_qs = [], [], [], []
    for _ in range(test_num):
        o = env.reset()
        old_state = save_state(env)
        with torch.torch.no_grad():
            q = get_value(o, md).numpy()
        orig_qs.append(q)
        _, r = run_extra_steps(env, o, 0, md, 1000, test_l)
        orig_rs.append(r)
        restore_state(env, old_state)
        for _ in range(adv_step_num):
            o = change_env_to_adv(env, md, init_state, adv_step_size, eps, sign)
        new_state = save_state(env)
        _, l2 = get_state_dis(new_state, old_state)
        l2_old.append(l2)
        _, l2 = get_state_dis(new_state, init_state)
        l2_init.append(l2)
        with torch.torch.no_grad():
            q = get_value(o, md).numpy()
        adv_qs.append(q)
        _, r = run_extra_steps(env, o, 0, md, 1000, test_l)
        adv_rs.append(r)
    print_2f(adv_step_num, get_mean(orig_rs), get_mean(adv_rs), get_mean(orig_qs), get_mean(adv_qs), "dis: ",  get_mean(l2_old), get_mean(l2_init))
        
def compute_loss_q(ac, ac_targ, data):
    gamma = 0.99
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
    q1 = ac.q1(o,a)
    q2 = ac.q2(o,a)
    with torch.no_grad():
        a2, _ = ac.pi(o2)
        q1_pi_targ = ac_targ.q1(o2, a2)
        q2_pi_targ = ac_targ.q2(o2, a2)
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
        backup = r + gamma * (1 - d) * (q_pi_targ)
    loss_q1 = ((q1 - backup)**2).mean()
    loss_q2 = ((q2 - backup)**2).mean()
    loss_q = loss_q1 + loss_q2
    return loss_q


def train_with_buffer(ac,  ac_targ, replay_buffer, q_optimizer):
    polyak=0.995
    batch_size = 32
    total_steps = (int)(600000 / batch_size)
    for t in range(total_steps):
        batch = replay_buffer.sample_batch(batch_size)
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(ac, ac_targ, batch)
        loss_q.backward()
        q_optimizer.step()
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

def get_random_action(o, md):
    return md.act(torch.as_tensor(o, dtype=torch.float32), False)
    

def testing(path, env_name, name, adv_step_num, adv_step_size, eps=None, sign=True):
    print(adv_step_num, adv_step_size, eps, sign)
    env = gym.make(env_name)
    init_state = get_init_state(env)    
    replay_size = int(1e6)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    print("get model")
    ac = get_models(path, env_name, name)
    print("use original q")
    test_model_adv(env, ac, adv_step_num, adv_step_size, eps, sign)
    max_ep_len = 100
    ac_targ = deepcopy(ac)
    for p in ac_targ.parameters():
        p.requires_grad = False
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_optimizer = Adam(q_params, lr=1e-3)
    total_steps = 3e6
    o, ep_ret, ep_len = env.reset(), 0, 0
    max_ep_len = 300
    rs = []
    for t in range(int(total_steps)):
        a = get_random_action(o, ac)
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        d = False if ep_len==max_ep_len else d
        replay_buffer.store(o, a, r, o2, d)
        o = o2
        if d or (ep_len == max_ep_len):
            rs.append(ep_ret)
            o, ep_ret, ep_len = env.reset(), 0, 0
            # rd = random.randint(0, adv_step_num)
            if(t < 3e5):
                rd = 0
            for _ in range(adv_step_num):
                o = change_env_to_adv(env, ac, init_state, adv_step_size, eps, sign)

        if (t+1) % 300000 == 0:
            print("start training at step ", t+1, get_mean(rs))
            rs = []
            train_with_buffer(ac, ac_targ, replay_buffer, q_optimizer)
            test_model_adv(env, ac, adv_step_num, adv_step_size, eps, sign)
        
        
testing('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_sac_base', 30, 0.01, 0.001, True)
testing('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_sac_base', 30, 0.01, 0.001, False)
# testing('/home/lclan/spinningup/data/', 'Walker2d-v3', 'Walker2d-v3_sac_base')
# testing('/home/lclan/spinningup/data/', 'Humanoid-v3', 'Humanoid-v3_sac_base')
# testing('/home/lclan/spinningup/data/', 'HalfCheetah-v3', 'HalfCheetah-v3_sac_base')
# testing('/home/lclan/spinningup/data/', 'Hopper-v3', 'Hopper-v3_sac_base')


# test_intial_state('/home/lclan/spinningup/data/', 'Walker2d-v3', 'Walker2d-v3_sac_base', [5e-3, 5e-2, 5e-1])
# test_intial_state('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_sac_base', [0.1, 1.0])
# test_intial_state('/home/lclan/spinningup/data/', 'Humanoid-v3', 'Humanoid-v3_sac_base', [1e-2, 0.1])
# test_intial_state('/home/lclan/spinningup/data/', 'HalfCheetah-v3', 'HalfCheetah-v3_sac_base', [0.1, 1.0, 2.0])
# test_intial_state('/home/lclan/spinningup/data/', 'Hopper-v3', 'Hopper-v3_sac_base', [5e-3, 5e-2, 5e-1])

# testing('/home/lclan/spinningup/data/', 'Walker2d-v3', 'Walker2d-v3_ppo_base', [5e-3])
# testing('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_ppo_base', [0.1])
# # testing('/home/lclan/spinningup/data/', 'Humanoid-v3', 'Humanoid-v3_ppo_base', [1e-2])
# # testing('/home/lclan/spinningup/data/', 'HalfCheetah-v3', 'HalfCheetah-v3_ppo_base', [0.1])
# # testing('/home/lclan/spinningup/data/', 'Hopper-v3', 'Hopper-v3_ppo_base', [5e-3])


# test_intial_state('/home/lclan/spinningup/data/', 'Walker2d-v3', 'Walker2d-v3_ppo_base', [5e-3, 5e-2, 5e-1])
# test_intial_state('/home/lclan/spinningup/data/', 'Ant-v3', 'Ant-v3_ppo_base', [0.1, 1.0])
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

# def set_state_with_obs(env, obs):
#     if(torch.is_tensor(obs)):
#         obs = obs.detach().numpy()
#     nq = env.model.nq
#     nv = env.model.nv
#     print(nq, nv, env.sim.data.qpos.shape, env.sim.data.qvel.shape, obs.shape)
#     qpos = obs[:nq]
#     qvel = obs[nq:nq+nv]
#     env.set_state(qpos, qvel)
#     new_obs = env.get_obs()
#     return new_obs
# def change_env_to_adv_w_grad(env, m1, max_eps_t):
#     o = env.get_obs()
#     with torch.set_grad_enabled(True):
#         oo = torch.as_tensor(o, dtype=torch.float32)
#         oo.requires_grad = True
#         q = get_value(oo, m1)
#         grad_o = torch.autograd.grad(q, [oo], retain_graph=True)[0]
#     grad_o = torch.clamp(grad_o.detach(), min=-1, max=1)
#     o_adv = oo - max_eps_t * grad_o.detach()
#     new_obs = set_state_with_obs(env, o_adv)
#     return new_obs
    

