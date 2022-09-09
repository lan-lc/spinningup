import os

try:
    os.remove(os.path.expanduser("~/miniconda/envs/py37/lib/python3.7/site-packages/mujoco_py/generated/mujocopy-buildlock.lock"))
except:
    pass

from spinup.utils.run_utils import ExperimentGrid
from spinup import gsac2_pytorch
import torch
from multiprocessing import Pool

def expr(args, seed):
    test_trajs_names = {}
    test_trajs_names['Walker2d-v3'] = './data/Walker2d-v3_sac_base_20_trajs.pkl'
    test_trajs_names['Hopper-v3'] = './data/Hopper-v3_sac_base_20_trajs.pkl'
    test_trajs_names['HalfCheetah-v3'] = './data/HalfCheetah-v3_sac_base_20_trajs.pkl'
    test_trajs_names['Ant-v3'] = './data/Ant-v3_sac_base_20_trajs.pkl'
    test_trajs_names['Humanoid-v3'] = './data/Humanoid-v3_sac_base_20_trajs.pkl'
    train_trajs_names = {}
    train_trajs_names['Walker2d-v3'] = './data/Walker2d-v3_sac_base_train_33_trajs.pkl'
    train_trajs_names['Hopper-v3'] = './data/Hopper-v3_sac_base_train_33_trajs.pkl'
    train_trajs_names['HalfCheetah-v3'] = './data/HalfCheetah-v3_sac_base_train_33_trajs.pkl'
    train_trajs_names['Ant-v3'] = './data/Ant-v3_sac_base_train_33_trajs.pkl'
    train_trajs_names['Humanoid-v3'] = './data/Humanoid-v3_sac_base_train_33_trajs.pkl'

    name = args.env + "_sac_gsac2_c40_e20"
    eg = ExperimentGrid(name)
    eg.add('env_name', args.env)
    eg.add('seed', [seed])
    eg.add('epochs', 900)
    eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', [(256, 256)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.ReLU], '')
    eg.add('test_trajs_name', test_trajs_names[args.env])
    eg.run(gsac2_pytorch, num_cpu=args.cpu)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=4)
    parser.add_argument('--env', type=str, default='Walker2d-v3')
    args = parser.parse_args()
    
    if args.num_runs > 1:
        # start multiprocessing only if more than one runs
        x = 1124
        with Pool(args.num_runs) as p:
            p.starmap(expr, [(args, seed) for seed in range(x, x + args.num_runs)])
    else:
        expr(args, 0)

