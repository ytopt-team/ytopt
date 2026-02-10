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

from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer

from ytopt.search.optimizer import Optimizer

# Parse comms, default options from commandline
nworkers, is_manager, libE_specs, user_args_in = parse_args()
num_sim_workers = nworkers - 1  # Subtracting one because one worker will be the generator

assert user_args_in, "learner, etc. not specified, e.g. --learner RF"
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
libE_specs['sim_dir_symlink_files'] = [here + f for f in ['kan.py', 'exe.pl', 'plopper.py', 'processexe.pl']]

# Declare the sim_f to be optimized, and the input/outputs
sim_specs = {
    'sim_f': init_obj,
    'in': ['p0', 'p1', 'p2'],
    'out': [('Object', float),('elapsed_sec', float),('Best', float)],
}

cs = ConfigurationSpace(seed=1234)
# M_GAUSSIANS 
p0 = Integer('p0', bounds=(0, 10), default=6)
# N_SIGMOIDS
p1 = Integer('p1', bounds=(0, 10), default=6)
# SPLINE_SMOOTH 
p2 = Float('p2', bounds=(0.0001, 0.1), default=0.001)

cs.add([p0, p1, p2])

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
    'out': [('p0', int, (1,)), ('p1', int, (1,)), ('p2', float, (1,)), ],
    'persis_in': sim_specs['in'] + ['Object'] + ['elapsed_sec'] + ['Best'],
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
    #H = H[H["returned"]]
    #dtypes = H[gen_specs['persis_in']].dtype
    #b = np.vstack(map(list, H[gen_specs['persis_in']]))
    #print(b)
    #np.savetxt('results.csv',b, header=','.join(dtypes.names), delimiter=',',fmt=','.join(['%s']*b.shape[1]))
