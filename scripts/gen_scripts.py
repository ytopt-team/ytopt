import os

HERE = os.path.dirname(os.path.abspath(__file__))
folder_name = ''

try:
    os.makedirs(f"{HERE}/{folder_name}")
except:
    pass

os.system(f"rm {HERE}/{folder_name}/*.sh")

benchmarks = ['ackley_real', 'ellipse_real', 'levy_real', 'perm_real', 'powersum_real', 
                'rosenbrock_real', 'schwefel_real',	'sum2_real', 'zakharov_real',
                'dixonprice_real', 'griewank_real',	'nesterov_real', 'powell_real',	
                'rastrigin_real', 'saddle_real', 'sphere_real',	'trid_real']

benchmarks = sorted(benchmarks)
print(benchmarks)
#benchmarks = ['ackley_real']

base_estimator = ['RF']#,'NND']
kappa = [0, 1.96]
acq_func= ['EI','LCB','PI','gp_hedge']
for benchmark_i in benchmarks:
    for base_estimator_i in base_estimator:                       
        for kappa_i in kappa:                       
            for acq_func_i in acq_func:
                exp_id = '{}_{}_{}_{}'.format(benchmark_i, base_estimator_i, kappa_i, acq_func_i)
                try:
                    os.makedirs(f"../experiments-01/{benchmark_i}")
                except:
                    pass
                with open(HERE+'/'+folder_name+'/'+exp_id+'.sh', "w+") as f:
                    f.write(f"mpiexec -np 2 python ../search/async-search.py "
                            f"--prob_dir=../benchmarks/{benchmark_i} "
                            f"--exp_dir=../experiments-01/{benchmark_i} "
                            f"--exp_id={exp_id} "
                            f"--max_evals=1000 "
                            f"--max_time=60 "
                            f"--base_estimator='{base_estimator_i}' "
                            f"--kappa={kappa_i} "
                            f"--acq_func='{acq_func_i}' \n")


base_estimator = ['DUMMY'] #,'NND']
kappa = [1.96]
acq_func= ['gp_hedge']
for benchmark_i in benchmarks:
    for base_estimator_i in base_estimator:                       
        for kappa_i in kappa:                       
            for acq_func_i in acq_func:
                exp_id = '{}_{}_{}_{}'.format(benchmark_i, base_estimator_i, kappa_i, acq_func_i)
                try:
                    os.makedirs(f"../experiments-01/{benchmark_i}")
                except:
                    pass
                with open(HERE+'/'+folder_name+'/'+exp_id+'.sh', "w+") as f:
                    f.write(f"mpiexec -np 2 python ../search/async-search.py "
                            f"--prob_dir=../benchmarks/{benchmark_i} "
                            f"--exp_dir=../experiments-01/{benchmark_i} "
                            f"--exp_id={exp_id} "
                            f"--max_evals=1000 "
                            f"--max_time=60 "
                            f"--base_estimator='{base_estimator_i}' "
                            f"--kappa={kappa_i} "
                            f"--acq_func='{acq_func_i}' \n")
