import numpy as np
from autotune import TuningProblem
from autotune.space import *
import os, sys, time, json, math
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from skopt.space import Real, Integer, Categorical
import csv, time 
from csv import writer
from csv import reader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE)+ '/plopper')
from plopper import Plopper
import pandas as pd
from sdv.tabular import GaussianCopula
from sdv.tabular import CopulaGAN
from sdv.evaluation import evaluate
from sdv.constraints import CustomConstraint, Between
import random, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_evals', type=int, default=10, help='maximum number of evaluations')
parser.add_argument('--n_refit', type=int, default=0, help='refit the model')
parser.add_argument('--seed', type=int, default=1234, help='set seed')
parser.add_argument('--top', type=float, default=0.1, help='how much to train')
parser.add_argument('--target', type=int, default=400, help='target task')
args = parser.parse_args()

MAX_EVALS   = int(args.max_evals)
N_REFIT     = int(args.n_refit)
TOP         = float(args.top)
RANDOM_SEED = int(args.seed)
TARGET_task = str(args.target)
print ('max_evals',MAX_EVALS, 'number of refit', N_REFIT, 'how much to train', TOP, 'seed', RANDOM_SEED, 'target task', TARGET_task)

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

Time_start = time.time()
print ('time...now', Time_start)

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/mmm_block_'+TARGET_task+'.cpp',dir_path)


x1=['BLOCK_SIZE']
def myobj(point: dict):
    def plopper_func(x):
        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        value = [point[x1[0]]]
        print('CONFIG:',point)
        params = ["BLOCK_SIZE"]
        result = obj.findRuntime(value, params, '/mmm_block_'+TARGET_task+'.cpp')
        return result

    x = np.array([point['BLOCK_SIZE']])
    results = plopper_func(x)
    print('OUTPUT:%f',results)
    return results

#### selet by best top x%   
X_opt = []
cutoff_p = TOP
print ('----------------------------- how much data to use?', cutoff_p) 
param_names = x1
n_param = len(param_names)
frames = []
for i_size in ['100','200','300']:#
    dataframe = pd.read_csv(dir_path+"/results_rf_"+str(i_size)+".csv")  
    dataframe['runtime'] = np.log(dataframe['objective']) # log(run time)
    dataframe['input']   = pd.Series(int(i_size) for _ in range(len(dataframe.index)))
    q_10_s = np.quantile(dataframe.runtime.values, cutoff_p)
    real_df = dataframe.loc[dataframe['runtime'] <= q_10_s]
    real_data = real_df.drop(columns=['elapsed_sec'])
    real_data = real_data.drop(columns=['objective'])
    frames.append(real_data)      
real_data   = pd.concat(frames)

constraint_input = Between(
    column='input',
    low=1,
    high=500,
    )

model = GaussianCopula(
            field_names = ['input','BLOCK_SIZE','runtime'],    
            field_transformers = {'input': 'integer',
                                  'BLOCK_SIZE': 'integer',
                                  'runtime': 'float'},
            constraints=[constraint_input],
            min_value =None,
            max_value =None
    )

filename = "results_sdv.csv"
fields   = ['BLOCK_SIZE','exe_time','predicted','elapsed_sec']
# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 

    evals_infer = []
    Max_evals = MAX_EVALS
    eval_master = 0
    while eval_master < Max_evals:         
        # update model
        model.fit(real_data)
        conditions = {'input': int(TARGET_task)}
        ss1 = model.sample(max(100,Max_evals),conditions=conditions)
        ss1 = ss1.drop_duplicates(subset='BLOCK_SIZE', keep="first")
        ss  = ss1.sort_values(by='runtime')#, ascending=False)
        new_sdv = ss[:Max_evals]
        max_evals = N_REFIT
        eval_update = 0
        stop = False
        while stop == False:
            for row in new_sdv.iterrows():
                if eval_update == max_evals:
                    stop = True
                    break    
                sample_point_val = row[1].values[1:]
                sample_point = {x1[0]:sample_point_val[0]}
                res          = myobj(sample_point)
                print (sample_point, res)
                evals_infer.append(res)
                now = time.time()
                elapsed = now - Time_start
                ss = [sample_point['BLOCK_SIZE']]+[res]+[sample_point_val[-1]]+[elapsed]
                csvwriter.writerow(ss)
                csvfile.flush()
                row_prev = row
                evaluated = row[1].values[1:]
                evaluated[-1] = float(np.log(res))
                evaluated = np.append(evaluated,row[1].values[0])
                real_data.loc[max(real_data.index)+1] = evaluated # real_data = [
                eval_update += 1
                eval_master += 1 
        
csvfile.close()           

