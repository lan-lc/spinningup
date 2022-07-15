import os

try:
    os.remove(os.path.expanduser("~/miniconda/envs/py37/lib/python3.7/site-packages/mujoco_py/generated/mujocopy-buildlock.lock"))
except:
    pass

from spinup.utils.run_utils import ExperimentGrid
from spinup import sac_pytorch
import torch
from multiprocessing import Pool

def expr(args, seed):

    name = args.env + '_' + args.name
    if args.id != '':
        name = name + '_' + str(args.id)      
    eg = ExperimentGrid(name)
    eg.add('env_name', args.env)
    eg.add('seed', [seed])
    eg.add('epochs', 900)
    eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', [(256, 256)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.ReLU], '')    
    eg.run(sac_pytorch, num_cpu=args.cpu)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=4)
    parser.add_argument('--env', type=str, default='Walker2d-v3')
    parser.add_argument('--name', type=str, default='sac_base')
    parser.add_argument('--id', type=str, default='')
    args = parser.parse_args()
    
    if args.num_runs > 1:
        # start multiprocessing only if more than one runs
        with Pool(args.num_runs) as p:
            p.starmap(expr, [(args, seed) for seed in range(args.num_runs)])
    else:
        expr(args, 0)