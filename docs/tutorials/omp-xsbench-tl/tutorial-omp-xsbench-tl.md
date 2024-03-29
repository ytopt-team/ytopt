Tutorial: Online tuning the OpenMP version of XSBench with transfer learning  
===================

This tutorial describes how to autotune ECP XSBench app in online tuning settings. 

We assume that you have checked out a copy of `ytopt` and install `sdv` from [https://github.com/sdv-dev/SDV](https://github.com/sdv-dev/SDV). For guidelines on how to get ytopt and sdv set up, refer [Install instructions](https://github.com/ytopt-team/ytopt/blob/online/README.md). 

Describing online tuning problem and our approach 
-----------------------

Autotuning refers to the process of generating a search space of possible implementations/configurations of a kernel or an application and evaluating a subset of implementations/configurations on a target platform through empirical measurements to identify the high-performing implementation/configuration. 

Most tuners are designed for offline tuning. However, tuning online by using the auto-tuner in production is highly needed. 

In the online tuning settings, how to gather prior knowledge on search space offline and leverage this on a target task real time are the key. To this end, we introduce a generative model that learns the prior information to autotuning a target task online. Procedure of our approach is as follows: 
- Gather performances of different configurations on source tasks offline 
- Given the online targe task, use a generative model to learn prior knowledge and suggest a promising configuration in production. 

In this tutorial, we present our approach to autotune ECP XSBench app `<https://github.com/ANL-CESAR/XSBench>`.

XSBench is a mini-app representing a key computational kernel of the Monte Carlo neutron transport algorithm [(reference)](https://github.com/ANL-CESAR/XSBench). Especially, it is the continuous energy macroscopic neutron cross section lookup kernel. XSBench supports the following command line options such as `-m` for Simulation method, `-g` for the number of gridpoints per nuclide, and `-m` for the number of Cross-section (XS) lookups. 

In this tutorual, we use different lookup sizes to define source offline and target online tasks. 

- We use the sizes of lookups of 100000, 1000000, 5000000 for offline tuning tasks and 10000000 for online tuning task.


Autotuning source tasks offline
-----------------------
First, we autotune source tasks offline based on `ytopt` bayesian optimization (BO) search with Random Forest surrogate model.

If you are not familiar with `ytopt` BO search. Please first follow a tutorial in <https://github.com/ytopt-team/ytopt/blob/online/docs/tutorials/omp-xsbench/tutorial-omp-xsbench.md>


Our search space contains three parameters: 1) `p0`: number of threads, 2) `p1`: block size for openmp dynamic schedule, 3) `p2`: turn on/off omp parallel.  


```python
# create an object of ConfigSpace 
cs = CS.ConfigurationSpace(seed=1234)
# number of threads
p0= CSH.UniformIntegerHyperparameter(name='p0', lower=2, upper=128, default_value=128)
#block size for openmp dynamic schedule
p1= CSH.OrdinalHyperparameter(name='p1', sequence=['10','20','40','64','80','100','128','160','200'], default_value='100')
#omp parallel
p2= CSH.CategoricalHyperparameter(name='p2', choices=["#pragma omp parallel for", " "], default_value=' ')
#add parameters to search space object
cs.add_hyperparameters([p0, p1, p2])
```

We define a search problem for each source task:
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/problem_s.py`
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/problem_m.py`  
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/problem_l.py`  

and an evaluating method (Plopper) for code generation and compilation:
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/plopper/plopper_s.py`
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/plopper/plopper_m.py`
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/plopper/plopper_l.py`

and a perl file to computes average the execution time:
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/exe_s.pl`  
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/exe_m.pl`  
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/exe_l.pl`  
 
 
<!-- [Source task 1 (s): 100000 lookups](https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/problem_s.py)  
[Source task 2 (m): 1000000 lookups](https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/problem_m.py) 
[Source task 3 (l): 5000000 lookups](https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/problem_l.py)  -->

We run the following command to autotune each task: 

Go to where the tuning problem is located such as
- `cd ytopt/benchmark/xsbench-omp-tl/xsbench`

And start search    
- `python -m ytopt.search.ambs --evaluator ray --problem problem_s.Problem --max-evals=100 --learner RF`
- `python -m ytopt.search.ambs --evaluator ray --problem problem_m.Problem --max-evals=100 --learner RF`
- `python -m ytopt.search.ambs --evaluator ray --problem problem_l.Problem --max-evals=100 --learner RF`

Once the search is finished, place the csv files in the same folder:
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/results_rf_s_xsbench.csv`
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/results_rf_m_xsbench.csv` 
- `https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/results_rf_l_xsbench.csv`

`https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/run.sh` is the run file to do all searches. 

Autotuning target task online
-----------------------

Now, we describe a standalone code to autotune a target task online `<https://github.com/ytopt-team/ytopt/blob/online/ytopt/benchmark/xsbench-omp-tl/xsbench/Run_online_TL.py>` 

It provides options: --max-evals flag sets the maximum number of evaluations, --n_refit flag sets how often to refit the generative model, --top flag sets how many of data from source tasks to use to train the generative model.

- Define the objective function `myobj` to evaluate a point in the search space.


```python
x1=['p0','p1','p2']
def myobj(point: dict):
    def plopper_func(x):
        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        value = [point[x1[0]],point[x1[1]],point[x1[2]]]
        print('CONFIG:',point)
        params = ["P0","P1","P2"]
        result = obj.findRuntime(value, params)
        return result
    x = np.array([point[f'p{i}'] for i in range(len(point))])
    results = plopper_func(x)
    print('OUTPUT:%f',results)
    return results
```

- Load data from source tasks. 


```python
X_opt = []
cutoff_p = TOP
param_names = x1
n_param = len(param_names)
frames = []
input_sizes = {}
input_sizes['s']  = [100000] 
input_sizes['m']  = [1000000]
input_sizes['l']  = [5000000]
input_sizes['xl'] = [10000000]

for i_size in ['s','m','l']:
    dataframe = pd.read_csv(dir_path+"/results_rf_"+str(i_size)+"_xsbench.csv")  
    dataframe['runtime'] = np.log(dataframe['objective']) # log(run time)
    dataframe['input']   = pd.Series(input_sizes[i_size][0] for _ in range(len(dataframe.index)))
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
    high=2100000000,
    )

model = GaussianCopula(
            field_names = ['input','p0','p1','p2','runtime'],    
            field_transformers = {'input': 'integer',
                                  'p0': 'categorical',
                                  'p1': 'categorical',
                                  'p2': 'categorical',
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
    conditions = {'input': input_sizes[TARGET_task][0]}
    ss1 = model.sample(max(1000,Max_evals),conditions=conditions)
    ss  = ss1.sort_values(by='runtime')
    new_kde = ss[:Max_evals]
    max_evals = N_REFIT
    eval_update = 0
    stop = False
    while stop == False:
        for row in new_kde.iterrows():
            if eval_update == max_evals:
                stop = True
                break    
            sample_point_val = row[1].values[1:]
            sample_point = {x1[0]:sample_point_val[0],
                        x1[1]:sample_point_val[1],
                        x1[2]:sample_point_val[2]}
            res          = myobj(sample_point)
            print (sample_point, res)
```

- Start search

`python Run_online_TL.py --max_evals 10 --n_refit 10 --target xl --top 0.3`

Look up the best configuration (found so far) and its value by inspecting the following created file: `results_sdv.csv`

The result shows search by and our model based online tuning. It shows tha our approach finds a high perfoming confiruation at the beginning of the search. 

<!-- ![xsbench tl](xsbench_tl.png) -->


```python

```
