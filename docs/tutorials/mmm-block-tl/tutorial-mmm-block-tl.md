Tutorial: Online tuning the block matrix multiplication with transfer learning
===================

This tutorial describes how to define autotuning problem and an evaluating method for autotuning the block matrix multiplication in online tuning settings. 

We assume that you have checked out a copy of `ytopt` and install `sdv` from [https://github.com/sdv-dev/SDV](https://github.com/sdv-dev/SDV). For guidelines on how to get ytopt and sdv set up, refer [Install instructions](https://github.com/ytopt-team/ytopt/blob/online/README.md). 

This example including the source code is borrowed from [http://opentuner.org/tutorial/gettingstarted/](http://opentuner.org/tutorial/gettingstarted/).

Describing online tuning problem and our approach 
-----------------------

Autotuning refers to the process of generating a search space of possible implementations/configurations of a kernel or an application and evaluating a subset of implementations/configurations on a target platform through empirical measurements to identify the high-performing implementation/configuration. 

Most tuners are designed for offline tuning. However, tuning online by using the auto-tuner in production is highly needed. 

In the online tuning settings, how to gather prior knowledge on search space offline and leverage this on a target task real time are the key. To this end, we introduce a generative model that learns the prior information to autotuning a target task online. Procedure of our approach is as follows: 
- Gather performances of different configurations on source tasks offline 
- Given the online targe task, use a generative model to learn prior knowledge and suggest a promising configuration in production. 

In this tutorial, we target to autotune the block size for matrix multiplication. Blocking is used to improve the temporal locality of inner loops such that data structures in a program are orgarnized into chunks, i.e. blocks (ref: [https://csapp.cs.cmu.edu/public/waside/waside-blocking.pdf](https://csapp.cs.cmu.edu/public/waside/waside-blocking.pdf)). We want to find the block size that gives the minimal execution time. 

In this tutorual, we use different block sizes to define source offline and target online tasks. 

- We use the sizes of lookups of (s)100, (m)200, (l)300 for offline tuning tasks and (target)500 for online tuning task.


Autotuning source tasks offline
-----------------------
First, we autotune source tasks offline based on `ytopt` bayesian optimization (BO) search with Random Forest surrogate model.

If you are not familiar with `ytopt` BO search. Please first follow a tutorial in <https://github.com/ytopt-team/ytopt/blob/online/docs/tutorials/mmm-block/tutorial-mmm-block.md>

Our search space contains one parameter; `BLOCK_SIZE`: number of blocks.  


```python
# create an object of ConfigSpace
cs = CS.ConfigurationSpace(seed=1234)
#block size for openmp dynamic schedule
p0= CSH.UniformIntegerHyperparameter(name='BLOCK_SIZE', lower=1, upper=100, default_value=5)
cs.add_hyperparameters([p0])
# problem space
input_space = cs
output_space = Space([Real(0.0, inf, name="time")])
```

We define a search problem for each source task:
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/mmm-block-tl/xsbench/problem_s.py`
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/mmm-block-tl/xsbench/problem_m.py`  
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/mmm-block-tl/xsbench/problem_l.py`  

and an evaluating method (Plopper) for code generation and compilation:
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/mmm-block-tl/plopper/plopper.py`

and a perl file to computes average the execution time:
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/mmm-block-tl/xsbench/exe.pl`
 
 
<!-- [Source task 1 (s): 100000 lookups](https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/problem_s.py)  
[Source task 2 (m): 1000000 lookups](https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/problem_m.py) 
[Source task 3 (l): 5000000 lookups](https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/problem_l.py)  -->

We run the following command to autotune each task: 

Go to where the tuning problem is located such as
- `cd ytopt/benchmark/mmm-block-tl/mmm-block`

And start search    
- `python -m ytopt.search.ambs --evaluator ray --problem problem_s.Problem --max-evals=30 --learner RF`
- `python -m ytopt.search.ambs --evaluator ray --problem problem_m.Problem --max-evals=30 --learner RF`
- `python -m ytopt.search.ambs --evaluator ray --problem problem_l.Problem --max-evals=30 --learner RF`

Once the search is finished, place the csv files in the same folder:
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/mmm-block-tl/mmm-block/results_rf_100.csv`
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/mmm-block-tl/mmm-block/results_rf_200.csv` 
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/mmm-block-tl/mmm-block/results_rf_300.csv`

`https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/mmm-block-tl/mmm-block/run.sh` is the run file to do all searches. 

Autotuning target task online
-----------------------

Now, we describe a standalone code to autotune a target task online `<https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/mmm-block-tl/mmm-block/Run_online_TL.py>` 

It provides options: --max-evals flag sets the maximum number of evaluations, --n_refit flag sets how often to refit the generative model, --top flag sets how many of data from source tasks to use to train the generative model.

- Define the objective function `myobj` to evaluate a point in the search space.


```python
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
        result = obj.findRuntime(value, params)
        return result

    x = np.array([point['BLOCK_SIZE']])
    results = plopper_func(x)
    print('OUTPUT:%f',results)
    return results
```

- Load data from source tasks. 


```python
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
```

- Define a generative model 


```python
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
```

- Fit the generative model and suggested configurations are evaluated


```python
Max_evals = MAX_EVALS
eval_master = 0
while eval_master < Max_evals:         
    # update model
    model.fit(real_data)
    conditions = {'input': TARGET_task}
    ss1 = model.sample(max(100,Max_evals),conditions=conditions)
    ss1 = ss1.drop_duplicates(subset='BLOCK_SIZE', keep="first")
    ss  = ss1.sort_values(by='runtime')
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
```

- Start search

`python Run_online_TL.py --max_evals 10 --n_refit 10 --target 500 --top 0.3`

Look up the best configuration (found so far) and its value by inspecting the following created file: `results_sdv.csv`

The result shows search by and our model based online tuning. It shows tha our approach finds a high perfoming confiruation at the beginning of the search. 

<!-- ![xsbench tl](xsbench_tl.png) -->

### Please check out another example of the OpenMP version of XSBench online 
- <https://github.com/ytopt-team/ytopt/blob/online/docs/tutorials/omp-xsbench-tl/tutorial-omp-xsbench-tl.md>


```python

```


```python

```
