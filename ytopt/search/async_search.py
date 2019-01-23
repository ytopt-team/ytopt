#!/usr/bin/env python
from __future__ import print_function
#from NeuralNetworksDropoutRegressor import NeuralNetworksDropoutRegressor
from mpi4py import MPI
import re
import os
import sys
import time
import json
import math
from skopt import Optimizer
import os
import argparse

from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
import numpy as np
from ytopt.search.NeuralNetworksDropoutRegressor import NeuralNetworksDropoutRegressor
from ytopt.search.search import Search
from ytopt.search.utils import tags, saveResults
seed = 1234

class AsyncSearch(Search):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        param_dict = kwargs
        self.acq_func = param_dict['acq_func']
        self.base_estimator=param_dict['base_estimator']
        self.kappa = param_dict['kappa']
        self.patience_fac = param_dict['patience_fac']

    @staticmethod
    def _extend_parser(parser):
        parser.add_argument('--base_estimator', action='store', dest='base_estimator',
                            nargs='?', type=str, default='RF',
                            help='which base estimator')
        parser.add_argument('--kappa', action='store', dest='kappa',
                            nargs='?', const=2, type=float, default='0',
                            help='kappa value')
        parser.add_argument('--acq_func', action='store', dest='acq_func',
                            nargs='?',  type=str, default='gp_hedge',
                            help='which acquisition function')
        parser.add_argument('--patience_fac', action='store', dest='patience_fac',
                            nargs='?', const=2, type=float, default='10',
                            help='patience_fac for early stopping; search stops when no improvement \
                            is seen for patience_fac * n evals')
        return parser

    def main(self):
        #  Initializations and preliminaries
        comm = MPI.COMM_WORLD   # get MPI communicator object
        size = comm.size        # total number of processes
        rank = comm.rank        # rank of this process
        status = MPI.Status()   # get MPI status object

        comm.Barrier()
        start_time = time.time()

        # Master process executes code below
        if rank == 0:
            num_workers = size - 1
            closed_workers = 0
            space = [self.spaceDict[key] for key in self.params]
            print("space: ", space)
            eval_counter = 0

            parDict = {}
            evalDict = {}
            resultsList = []
            parDict['kappa']=self.kappa
            init_x = []
            delta = 0.05
            #patience = max(100, 3 * num_workers-1)
            patience = len(self.params) * self.patience_fac
            last_imp = 0
            curr_best = math.inf

            if self.base_estimator =='NND':
                opt = Optimizer(space, base_estimator=NeuralNetworksDropoutRegressor(), acq_optimizer='sampling',
                            acq_func = self.acq_func, acq_func_kwargs=parDict, random_state=seed)
            else:
                opt = Optimizer(space, base_estimator=self.base_estimator, acq_optimizer='sampling',
                            acq_func=self.acq_func, acq_func_kwargs=parDict, random_state=seed)
            print('Master starting with {} workers'.format(num_workers))

            while closed_workers < num_workers:
                data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                source = status.Get_source()
                tag = status.Get_tag()
                elapsed_time = float(time.time() - start_time)
                print('elapsed_time:%1.3f'%elapsed_time)
                if tag == tags.READY:
                    if last_imp < patience and eval_counter < self.max_evals and elapsed_time < self.max_time:
                        if self.starting_point is not None:
                            x = self.starting_point
                            if num_workers-1 > 0:
                                init_x = opt.ask(n_points=num_workers-1)
                            self.starting_point = None
                        else:
                            if len(init_x) > 0:
                                x = init_x.pop(0)
                            else:
                                x = opt.ask(n_points=1, strategy='cl_min')[0]
                        key = str(x)
                        print('sample %s' % key)
                        if key in evalDict.keys():
                            print('%s already evalauted' % key)
                        evalDict[key] = None
                        task = {}
                        task['x'] = x
                        task['eval_counter'] = eval_counter
                        task['rank_master'] = rank
                        #task['start_time'] = elapsed_time
                        print('Sending task {} to worker {}'.format (eval_counter, source))
                        comm.send(task, dest=source, tag=tags.START)
                        eval_counter = eval_counter + 1
                    else:
                        comm.send(None, dest=source, tag=tags.EXIT)
                elif tag == tags.DONE:
                    result = data
                    result['end_time'] = elapsed_time
                    print('Got data from worker {}'.format(source))
                    resultsList.append(result)
                    x = result['x']
                    y = result['cost']
                    opt.tell(x, y)
                    percent_improv = -100*((y+0.1) - (curr_best+0.1))/(curr_best+0.1)
                    if y < curr_best:
                        if percent_improv >= delta or curr_best==math.inf:
                            curr_best = y
                            last_imp = 0
                    else:
                        last_imp = last_imp+1
                    print('curr_best={} percent_improv={} patience={}/{}'.format(curr_best, percent_improv, last_imp, patience))
                elif tag == tags.EXIT:
                    print('Worker {} exited.'.format(source))
                    closed_workers = closed_workers + 1
                    resultsList = data
                    print('Search finished..')
                    #resultsList = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status) #comm.recv(source=MPI.ANY_SOURCE, tag=tags.EXIT)
                    #print(resultsList)
                    saveResults(resultsList, self.results_json_fname, self.results_csv_fname)
                    y_best = np.min(opt.yi)
                    best_index = np.where(opt.yi==y_best)[0][0]
                    x_best = opt.Xi[best_index]
                    print('Best: x = {}; y={}'.format(y_best, x_best))
        else:
            # Worker processes execute code below
            name = MPI.Get_processor_name()
            print("worker with rank %d on %s." % (rank, name))
            resultsList = []
            while True:
                comm.send(None, dest=0, tag=tags.READY)
                task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()
                if tag == tags.START:
                    result = self.evaluate(self.problem, task, self.jobs_dir, self.results_dir)
                    elapsed_time = float(time.time() - start_time)
                    result['elapsed_time'] = elapsed_time
                    print(result)
                    resultsList.append(result)
                    comm.send(result, dest=0, tag=tags.DONE)
                elif tag == tags.EXIT:
                    print(f'Exit rank={comm.rank}')
                    break
            comm.send(resultsList, dest=0, tag=tags.EXIT)

if __name__ == "__main__":
    args = AsyncSearch.parse_args()
    search = AsyncSearch(**vars(args))
    search.main()