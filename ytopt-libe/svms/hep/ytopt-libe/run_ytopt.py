"""
Runs libEnsemble to call the ytopt ask/tell interface in a generator function,
and the ytopt findRunTime interface in a simulator function.

Execute locally via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python run_ytopt_xsbench.py
   python run_ytopt_xsbench.py --nworkers 3 --comms local

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

import os
import glob
import secrets
import numpy as np
import time

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

from ytopt_obj import init_obj  # Simulator function, calls Plopper
from ytopt_asktell import persistent_ytopt  # Generator function, communicates with ytopt optimizer

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import ConfigurationSpace, EqualsCondition
from ytopt.search.optimizer import Optimizer

# Parse comms, default options from commandline
nworkers, is_manager, libE_specs, user_args_in = parse_args()
num_sim_workers = nworkers - 1  # Subtracting one because one worker will be the generator

assert len(user_args_in), "learner, etc. not specified, e.g. --learner RF"
user_args = {}
for entry in user_args_in:
    if entry.startswith('--'):
        if '=' not in entry:
            key = entry.strip('--')
            value = user_args_in[user_args_in.index(entry)+1]
        else:
            split = entry.split('=')
            key = split[0].strip('--')
            value = split[1]

    user_args[key] = value

req_settings = ['learner','max-evals']
assert all([opt in user_args for opt in req_settings]), \
    "Required settings missing. Specify each setting in " + str(req_settings)

# Set options so workers operate in unique directories
here = os.getcwd() + '/'
libE_specs['use_worker_dirs'] = True
libE_specs['sim_dirs_make'] = False  # Otherwise directories separated by each sim call
libE_specs['ensemble_dir_path'] = './ensemble_' + secrets.token_hex(nbytes=4)

# Copy or symlink needed files into unique directories
libE_specs['sim_dir_symlink_files'] = [here + f for f in ['dlp.py', 'exe.pl', 'plopper.py', 'processexe.pl']]

# Declare the sim_f to be optimized, and the input/outputs
sim_specs = {
    'sim_f': init_obj,
    'in': ['p0', 'p1', 'p2', 'p3','p4'],
    'out': [('RUNTIME', float),('elapsed_sec', float)],
}

cs = CS.ConfigurationSpace(seed=1234)

#p0= CSH.UniformFloatHyperparameter(name='p0', lower=0.0, upper=0.001, q=0.00001, default_value=0.0001, log = False) #mix
p0= CSH.UniformFloatHyperparameter(name='p0', lower=0.0, upper=0.001, q=0.00001, log = False) #mix

#p1= CSH.UniformFloatHyperparameter(name='p1', lower=0.00001, upper=0.001, q=0.00001, default_value=0.0001,log = False) #sig_ratio
p1= CSH.UniformFloatHyperparameter(name='p1', lower=0.00001, upper=0.001, q=0.00001, log = False) #sig_ratio

p2= CSH.UniformFloatHyperparameter(name='p2', lower=0.9, upper=1.5, q=0.001,default_value=1,  log = False) #gaus_ratio

p3= CSH.UniformFloatHyperparameter(name='p3', lower=-1.0, upper=1.0, q=0.01, default_value=0.0, log = False) #coef0

p4= CSH.UniformFloatHyperparameter(name='p4', lower=0.37, upper=10, q=0.01, default_value=1, log = False) #C

cs.add_hyperparameters([p0, p1, p2, p3,p4])
#cond = EqualsCondition(p3, p4, "mixed_kernel")
#cs.add_conditions([cond])

ytoptimizer = Optimizer(
    num_workers=num_sim_workers,
    space=cs,
    learner=user_args['learner'],
    liar_strategy='cl_max',
    acq_func='gp_hedge',
    set_KAPPA=1.96,
    set_SEED=2345,
    set_NI=10,
)

# Declare the gen_f that will generate points for the sim_f, and the various input/outputs
gen_specs = {
    'gen_f': persistent_ytopt,
    'out': [('p0', float, (1,)), ('p1', float, (1,)),('p2', float, (1,)),('p3', float, (1,)),('p4', float, (1,))],
    'persis_in': sim_specs['in'] + ['RUNTIME'] + ['elapsed_sec'],
    'user': {
        'ytoptimizer': ytoptimizer,  # provide optimizer to generator function
        'num_sim_workers': num_sim_workers,
    },
}

alloc_specs = {
    'alloc_f': alloc_f,
    'user': {'async_return': True},
}

# Specify when to exit. More options: https://libensemble.readthedocs.io/en/main/data_structures/exit_criteria.html
exit_criteria = {'gen_max': int(user_args['max-evals'])}

# Added as a workaround to issue that's been resolved on develop
persis_info = add_unique_random_streams({}, nworkers + 1)

# Perform the libE run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            alloc_specs=alloc_specs, libE_specs=libE_specs)

# Save History array to file
if is_manager:
    print("\nlibEnsemble has completed evaluations.")
    #save_libE_output(H, persis_info, __file__, nworkers)

    #print("\nSaving just sim_specs[['in','out']] to a CSV")
    #H = np.load(glob.glob('*.npy')[0])
    #H = H[H["sim_ended"]]
    #dtypes = H[gen_specs['persis_in']].dtype
    #b = np.vstack(map(list, H[gen_specs['persis_in']]))
    #print(b)
    #np.savetxt('results.csv',b, header=','.join(dtypes.names), delimiter=',',fmt=','.join(['%s']*b.shape[1]))
