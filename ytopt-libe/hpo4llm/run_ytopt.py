"""
Runs libEnsemble to call the ytopt ask/tell interface in a generator function,
and the ytopt findRunTime interface in a simulator function.

Execute locally via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python run_ytopt_xsbench.py
   python run_ytopt_xsbench.py --nworkers 3 --comms local

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

import os
import sys
import glob
import secrets
import numpy as np
import time

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

# sys.path.insert(0, os.path.dirname(__file__))
print(os.getcwd())
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
#libE_specs['sim_dir_symlink_files'] = [here + f for f in ['dlp.py', 'exe.pl', 'plopper.py', 'processexe.pl','trained_model.pt']]
libE_specs['sim_dir_symlink_files'] = [here + f for f in ['dlp.py', 'exe.pl', 'plopper.py', 'processexe.pl']]

# Declare the sim_f to be optimized, and the input/outputs
sim_specs = {
    'sim_f': init_obj,
    'in': ['p0','p1','p2','p3','p4','p5','p6','p7','p8','p9'], 
    'out': [('objective', float),('elapsed_sec', float)],
}

cs = CS.ConfigurationSpace(seed=1234)

# Hoffman coefficients
# #e
# p0 = CSH.UniformFloatHyperparameter(name='p0', lower=1.6, upper=1.7, default_value=1.69, q=0.01)
# #a
# p1 = CSH.UniformFloatHyperparameter(name='p1', lower=350, upper=450, default_value=406.4, q=0.1)
# #b
# p2 = CSH.UniformFloatHyperparameter(name='p2', lower=350, upper=450, default_value=410.7, q=0.1)
# #alpha
# p3 = CSH.UniformFloatHyperparameter(name='p3', lower=0.3, upper=0.5, default_value=0.34, q=0.01)
# #beta
# p4 = CSH.UniformFloatHyperparameter(name='p4', lower=0.2, upper=0.5, default_value=0.28, q=0.01)

# Frantar coefficients

# #e
# p0 = CSH.UniformFloatHyperparameter(name='p0', lower=0.6, upper=0.7, default_value=0.651, q=0.001)
# #_as
# p1 = CSH.UniformFloatHyperparameter(name='p1', lower=15.0, upper=18, default_value=16.8, q=0.1)
# #bs
# p2 = CSH.UniformFloatHyperparameter(name='p2', lower=0.6, upper=0.8, default_value=0.722, q=0.001)
# #cs
# p3 = CSH.UniformFloatHyperparameter(name='p3', lower=30, upper=50, default_value=45, q=1)
# #bn
# p4 = CSH.UniformFloatHyperparameter(name='p4', lower=0.1, upper=0.3, default_value=0.245, q=0.001)
# #ad
# p5 = CSH.UniformFloatHyperparameter(name='p5', lower=600000000, upper=750000000, default_value=690000000, q=1)
# #bd
# p6 = CSH.UniformFloatHyperparameter(name='p6', lower=0.1, upper=0.3, default_value=0.203, q=0.001)

# Abnar coefficients

#e
p0 = CSH.UniformFloatHyperparameter(name='p0', lower=0.8, upper=1, default_value=0.94, q=0.01)
#a
p1 = CSH.UniformFloatHyperparameter(name='p1', lower=16000, upper=17000, default_value=16612.50, q=0.01)
#b
p2 = CSH.UniformFloatHyperparameter(name='p2', lower=5300, upper=5500, default_value=5455.67, q=0.01)
#c
p3 = CSH.UniformFloatHyperparameter(name='p3', lower=0.4, upper=0.5, default_value=0.4598, q=0.0001)
#d
p4 = CSH.UniformFloatHyperparameter(name='p4', lower=15, upper=20, default_value=17.26, q=0.01)
#alpha
p5 = CSH.UniformFloatHyperparameter(name='p5', lower=0.4, upper=0.8, default_value=0.5962, q=0.0001)
#beta
p6 = CSH.UniformFloatHyperparameter(name='p6', lower=0.2, upper=0.6, default_value=0.3954, q=0.0001)
#lambda
p7 = CSH.UniformFloatHyperparameter(name='p7', lower=-.2, upper=0.2, default_value=-0.1666, q=0.0001)
#delta
p8 = CSH.UniformFloatHyperparameter(name='p8', lower=0.1, upper=0.2, default_value=0.1603, q=0.0001)
#gamma
p9 = CSH.UniformFloatHyperparameter(name='p9', lower=0.1, upper=0.2, default_value=0.1595, q=0.0001)

#cs.add_hyperparameters([p0, p1, p2, p3, p4])
#cs.add_hyperparameters([p0, p1, p2, p3, p4, p5, p6])
cs.add_hyperparameters([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9])

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
    'out': [('p0', float, (1,)), ('p1', float, (1,)), ('p2', float, (1,)), ('p3', float, (1,)), ('p4', float, (1,)), ('p5', float, (1,)), ('p6', float, (1,)), ('p7', float, (1,)), ('p8', float, (1,)), ('p9', float, (1,))],
    'persis_in': sim_specs['in'] + ['objective'] + ['elapsed_sec'],
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
