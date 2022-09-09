from cgi import test
from copy import deepcopy
from copyreg import pickle
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.gsac.core as core
from spinup.utils.logx import EpochLogger
import pickle
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

class CheckPoint:
    def __init__(self, state, future_ret, ep_len, obs, epoch, old_ret, future_steps_num):
        self.state = state
        self.future_ret = future_ret
        self.future_steps_num = future_steps_num
        self.ep_len = ep_len
        self.o = obs
        self.epoch = epoch
        self.old_ret = old_ret
        
    def get_save_score(self):
        return self.future_ret  # + ((self.old_ret / self.ep_len)* self.future_steps_num)
    
    def get_q(self, o, a, ac):
        o = torch.as_tensor(o, dtype=torch.float32)
        a = torch.as_tensor(a, dtype=torch.float32)
        with torch.no_grad():
            q1 = ac.q1(o, a)
            q2 = ac.q2(o, a)
            return torch.min(q1, q2).detach().numpy()
    
    def get_sample_score(self, ac, sample_rule):
        a = ac.act(torch.as_tensor(self.o, dtype=torch.float32), True)
        v = self.get_q(self.o, a, ac)
        if sample_rule == 0:
            return -v
        elif sample_rule == 1:     
            return self.future_ret  * 100.0 / self.future_steps_num - v
        return 0

def gsac2(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, test_trajs_name=None, train_trajs_top_ratio=0.5, 
        trajs_sample_ratio = 0.5, sample_num = 1, continue_step = 100, sample_rule = 0, 
        future_ret_step_num = -1, over_write_ratio = 1.,
        ):
    """
    Soft Actor-Critic (SAC)
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:
            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================
            Calling ``pi`` should return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs to run and train agent.
        replay_size (int): Maximum length of replay buffer.
        gamma (float): Discount factor. (Always between 0 and 1.)
        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:
            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)
        lr (float): Learning rate (used for both policy and value learning).
        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)
        batch_size (int): Minibatch size for SGD.
        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.
        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.
        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.
        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """
    
    if future_ret_step_num  == -1:
        future_ret_step_num  = int(continue_step / 2)
    
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    print("start training gsac with rns = ", env.get_reset_noise_scale())
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    
    test_trajs = None
    if test_trajs_name != None:        
        with open(test_trajs_name, 'rb') as f:
            test_trajs = pickle.load(f)
        print(test_trajs_name, " has total ", len(test_trajs), " trajs")
    
    
    
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)
    
    def get_random_init_state(trajs, extra_step_num):
        while True:
            tmp = trajs[random.randint(0, len(trajs)-1)]
            if tmp[0] + extra_step_num < 990:
                return tmp
    
    def get_q(o, a):
        o = torch.as_tensor(o, dtype=torch.float32)
        a = torch.as_tensor(a, dtype=torch.float32)
        with torch.no_grad():
            q1 = ac.q1(o, a)
            q2 = ac.q2(o, a)
            return torch.min(q1, q2).detach().numpy()
    
    def get_v(o):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32), True)
        return get_q(o, a)
        
    def restore_state(env, old_state):
        env.reset()
        env.sim.set_state(old_state)
        env.sim.forward()
        return env.get_obs()
    
    def save_state(env):
        return env.sim.get_state()
    
        
    def sample_checkpoint_id_by_rule(sample_rule):
        best_id = random.randint(0, len(check_points)-1)
        best_score = check_points[best_id].get_sample_score(ac, sample_rule)
        for _ in range(1, sample_num):
            id = random.randint(0, len(check_points)-1)
            score = check_points[id].get_sample_score(ac, sample_rule)
            if score > best_score:
                best_score = score
                best_id = id
        logger.store(BestSampleScore=best_score)
        logger.store(OldReturn=check_points[best_id].future_ret)
        return best_id, best_score
    
    def run_extra_steps(env, o, ep_len, max_ep_len, step_num):
        # return 0
        # print(env.done)
        total_r = 0
        l = 0
        for i in range(step_num):
            l +=1
            o, r, d, _ = env.step(get_action(o, True))
            total_r += r
            ep_len += 1
            if d or (ep_len == max_ep_len):
                # print(i, d, r, ep_len, max_ep_len)
                return d, total_r, l
        return d, total_r, l
    
    avg_test_ret = 0
    best_test_avg = 0
    avg_testG_done = 0
    best_testG_avg = 1.
    def test_agent():
        nonlocal avg_test_ret
        nonlocal best_test_avg
        nonlocal avg_testG_done
        nonlocal best_testG_avg
        ratio = 30
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            avg_test_ret *= (ratio-1) / ratio
            avg_test_ret += ep_ret
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        if best_test_avg * 1.01 < avg_test_ret:
            best_test_avg = avg_test_ret
            print("best model test avg: ", best_test_avg/ratio)
            logger.save_state({'env': env}, itr=0)
        logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        extra_step_num = 500
        if test_trajs != None:
            ratio *= 2
            for j in range(num_test_episodes*2):
                i, old_state = get_random_init_state(test_trajs, 500)
                o  = restore_state(test_env, old_state)
                dd, rr, ll = run_extra_steps(test_env, o, i, 1000, 500)
                if(rr < extra_step_num):
                    dd = True
                avg_testG_done *= (ratio-1) / ratio
                avg_testG_done += float(dd)
                logger.store(TestGEpRet=rr, TestGDoneMean = float(dd),  TestGEpLen=float(ll))
            if best_testG_avg > avg_testG_done * 1.01:
                best_testG_avg = avg_testG_done
                print("best model testG avg: ", best_testG_avg/ratio)
                logger.save_state({'env': env}, itr=1)
    
    check_points = []
    
    avg_rets = [1]*40
    avg_rets_current_id = 0
    
    
   
    
    # trajs[0] = state, o, r, ep_len
    def update_check_points(trajs, old_id, epoch):
        nonlocal check_points
        nonlocal avg_rets
        nonlocal avg_rets_current_id
        rs = [x[2] for x in trajs]
        crs = [0] * (len(rs)+1)
        for i in range(len(rs)):
            crs[i+1] = crs[i]+ rs[i]
        total_ret = sum(rs)
        old_ret = 0
        if old_id >= 0:
            old_ret = check_points[old_id].old_ret
            new_future_ret = sum(rs[:future_ret_step_num])
            if check_points[old_id].future_ret < new_future_ret:
                check_points[old_id].future_ret = new_future_ret
        avg_ret = total_ret / len(rs)
        x = deepcopy(avg_rets)
        x.sort()
        thr = x[int(len(x) * (1 - train_trajs_top_ratio) )]
        if avg_ret >= thr:
            usable_num = len(trajs) - future_ret_step_num  - 12
            if usable_num > 0:
                ids = random.sample(list(range(10, len(trajs) - future_ret_step_num  -1)), 
                                    int(len(trajs) / future_ret_step_num ))
                for id in ids:
                    future_ret = crs[id+future_ret_step_num +1] - crs[id+1]
                    if future_ret/future_ret_step_num >= thr:
                        # state, future_ret, ep_len, obs, epoch, old_ret, future_steps_num
                        cp = CheckPoint(trajs[id][0], future_ret, trajs[id][3], trajs[id][1], 
                                        epoch, old_ret + crs[id+1], future_ret_step_num )
                        if len(check_points) < 2000:
                            check_points.append(cp)
                        else:
                            cps_ids = random.sample(list(range(len(check_points))), 20)
                            lowest_id = cps_ids[0]
                            lowest_future_ret = check_points[lowest_id].future_ret
                            for cps_id in cps_ids:
                                future_ret = check_points[cps_id].future_ret
                                if future_ret < lowest_future_ret:
                                    lowest_id = cps_id
                                    lowest_future_ret = check_points[lowest_id].future_ret
                            if cp.future_ret > lowest_future_ret * over_write_ratio:
                                check_points[lowest_id] = cp
        if avg_ret > 1:
            avg_rets[avg_rets_current_id] = avg_ret
            avg_rets_current_id += 1
            avg_rets_current_id = avg_rets_current_id % len(avg_rets)
        

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    max_step_remain = 1000
    trajs = [(save_state(env), o, 0, ep_len)]
    old_id = -1
    
    # Main loop: collect experience in env and update/log each epoch
    
    for t in range(total_steps):
        max_step_remain -= 1
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        trajs.append((save_state(env), o, r, ep_len))

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len) or (max_step_remain == 0):
            if len(check_points) >= 1:
                sample_checkpoint_id_by_rule(sample_rule)
            else:
                logger.store(BestSampleScore=0)
                logger.store(OldReturn=0)
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            epoch = (t+1) // steps_per_epoch
            update_check_points(trajs, old_id, epoch)
            o, ep_ret, ep_len = env.reset(), 0, 0
            max_step_remain = 1000
            trajs = [(save_state(env), o, 0, ep_len)]
            old_id = -1

            if trajs_sample_ratio < 0.001:
                ratio = -1
            else:
                ratio = 1000. / float(1000. + (1./trajs_sample_ratio-1.)*continue_step)
            
            if random.random() < ratio:
                if len(check_points) > 500:
                    max_step_remain = continue_step
                    id, _ = sample_checkpoint_id_by_rule(sample_rule)
                    cp = deepcopy(check_points[id])
                    ep_ret = cp.old_ret
                    ep_len = cp.ep_len
                    old_id = id
                    o = restore_state(env, cp.state)
                    trajs = [(save_state(env), o, 0, ep_len)]
            
            logger.store(PureRatio=ratio)
            
                

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()
            
            # Save Checkpoints Info
            logger.store(CPLen=len(check_points))
            avg_epochs = 0
            avg_future_ret = 0
            for cp in check_points:
                avg_epochs += cp.epoch
                avg_future_ret += cp.future_ret
            if len(check_points) > 2:
                logger.store(CPEpoch = avg_epochs / len(check_points))
                logger.store(CPFutureRet = avg_future_ret / len(check_points))
            else:
                logger.store(CPEpoch = 0)
                logger.store(CPFutureRet = 0)
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('BestSampleScore', average_only=True)
            logger.log_tabular('OldReturn', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.log_tabular('TestGEpRet', average_only=True)
            logger.log_tabular('TestGDoneMean', average_only=True)
            logger.log_tabular('TestGEpLen', average_only=True)
            logger.log_tabular('PureRatio', average_only=True)
            logger.log_tabular('CPLen', average_only=True)
            logger.log_tabular('CPEpoch', average_only=True)
            logger.log_tabular('CPFutureRet', average_only=True)
            
            
            
            logger.dump_tabular()
            print(logger.output_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    gsac2(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)