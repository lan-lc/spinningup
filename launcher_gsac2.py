import os

from traitlets import default

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
    test_trajs_names['Walker2d-v3'] = './data/Walker2d-v3_sac_base_50_trajs.pkl'
    test_trajs_names['Hopper-v3'] = './data/Hopper-v3_sac_base_50_trajs.pkl'
    test_trajs_names['HalfCheetah-v3'] = './data/HalfCheetah-v3_sac_base_50_trajs.pkl'
    test_trajs_names['Ant-v3'] = './data/Ant-v3_sac_base_50_trajs.pkl'
    test_trajs_names['Humanoid-v3'] = './data/Humanoid-v3_sac_base_50_trajs.pkl'

    name = args.env + "_" + args.name
    eg = ExperimentGrid(name)
    eg.add('env_name', args.env)
    eg.add('seed', [seed])
    eg.add('epochs', args.epochs)
    eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', [(256, 256)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.ReLU], '')
    eg.add('test_trajs_name', test_trajs_names[args.env])
    eg.add('train_trajs_top_ratio', args.train_trajs_top_ratio)
    eg.add('trajs_sample_ratio', args.trajs_sample_ratio)
    eg.add('sample_num', args.sample_num)
    eg.add('continue_step', args.continue_step)
    eg.add('sample_rule', args.sample_rule)
    eg.add('over_write_ratio', args.over_write_ratio)
    
    eg.run(gsac2_pytorch, num_cpu=args.cpu)

def all_expr(args, seed):
    if args.env == 'three' or args.env == 'all':
        args.env = 'Walker2d-v3'
        expr(args,seed)
        args.env = 'Humanoid-v3'
        expr(args,seed)
        args.env = 'Hopper-v3'
        expr(args,seed)
        if args.env == 'all':
            args.env = 'Ant-v3'
            expr(args,seed)
            args.env = 'HalfCheetah-v3'
            expr(args,seed)
    else:
        expr(args,seed)
        
def set_default_and_name(args):
    name = args.name
    if args.epochs == None:
        if args.env == 'Hopper-v3':
            args.epochs = 265
        else:
            args.epochs = 800
    if args.train_trajs_top_ratio == None:
        args.train_trajs_top_ratio = 0.5
    else:
        name += '_tttr' + (str(int(args.train_trajs_top_ratio*100)))
    
    if args.trajs_sample_ratio == None:
        args.trajs_sample_ratio = 0.5
    else:
        name += '_tsr' + (str(int(args.trajs_sample_ratio * 10)))
    
    if args.sample_num == None:
        args.sample_num = 1
    else:
        name += '_sn' + (str(args.sample_num))
        
    if args.continue_step == None:
        args.continue_step = 100
    else:
        name += '_cs' + (str(args.continue_step))
    
    if args.sample_rule == None:
        args.sample_rule = 0
    else:
        name += '_sr' + (str(args.sample_rule)) 
    
    if args.over_write_ratio == None:
        args.over_write_ratio = 1.
    else:
        name += '_owr' + (str( int(args.over_write_ratio*10) )) 
    
    if args.start_sample_ratio == None:
        args.start_sample_ratio = 0.5
    else:
        name += "_ssr" + (str(int(args.start_sample_ratio*100)))
    
    args.name = name
    return name
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--env', type=str, default='Walker2d-v3')
    parser.add_argument('--name', type=str, default='gsac3')
    parser.add_argument('--seed', type=int, default=1324)
    parser.add_argument('--epochs', type=int, default=900)
    
    parser.add_argument('--train_trajs_top_ratio', '-tttr', type=float)
    parser.add_argument('--trajs_sample_ratio', '-tsr', type=float)
    parser.add_argument('--sample_num', '-sn', type=int)
    parser.add_argument('--continue_step', '-cs', type=int)
    parser.add_argument('--sample_rule', '-sr', type=int)
    parser.add_argument('--over_write_ratio', '-owr', type=float)
    parser.add_argument('--start_sample_ratio', '-ssr', typ=float)

    
    args = parser.parse_args()
    set_default_and_name(args)
    print(args)
    print("name ", args.name)
    
    if args.num_runs > 1:
        # start multiprocessing only if more than one runs
        x = args.seed
        with Pool(args.num_runs) as p:
            p.starmap(all_expr, [(args, seed) for seed in range(x, x + args.num_runs)])
    else:
        all_expr(args, args.seed)

