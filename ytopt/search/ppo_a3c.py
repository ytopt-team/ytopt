#!/usr/bin/env python
from __future__ import print_function
from mpi4py import MPI
import re
import os
import sys
import time
import json
import math
import os
import argparse

import numpy as np

from ytopt.search.search import Search
from ytopt.search.utils import tags, saveResults
from ppo.gym_ytopt.agent.train import Train

from ppo.gym_ytopt.agent.lstm_policy import policy_fn

class PpoA3c(Search):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        param_dict = kwargs
        param_dict['base_estimator'] = 'A3C' 
        self.num_agents = param_dict['num_agents']
        self.base_estimator = param_dict['base_estimator']

    @staticmethod
    def _extend_parser(parser):
        parser.add_argument('--base_estimator', action='store', dest='base_estimator',
                            nargs='?', type=str, default='PPO',
                            help='which base estimator')
        parser.add_argument('--num_agents', type=int, default=1, help='number of parallel agents for A3C')
        return parser

    def main(self):
        os.environ['OPENAI_LOGDIR'] = os.path.abspath(self.exp_dir)

        # Initializations and preliminaries
        comm = MPI.COMM_WORLD   # get MPI communicator object
        size = comm.size        # total number of processes
        rank = comm.rank        # rank of this process
        status = MPI.Status()   # get MPI status object

        eval_counter = 0

        assert self.num_agents >= 1, "You must have at least one agent for the search."
        assert self.num_agents < size, "You must have available processes to compute your evaluations."
        comm.Barrier()
        start_time = time.time()

        if rank < self.num_agents:    
            # AGENTS
            print(f'[A, r={rank}] starting')
            # workers linked with current agent
            rank_workers = [i for i in range(size-self.num_agents+rank, size, self.num_agents)]
            print(f'[A, r={rank}] workers: {rank_workers}')
            group_world = comm.Get_group()
            group_agents = MPI.Group.Incl(group_world, [i for i in range(self.num_agents)])
            newcomm = comm.Create_group(group_agents)
            trainer = Train(self.problem, rank_workers, policy_fn, comm=newcomm, tags=tags, max_time=self.max_time)
            trainer.train()
            #resultsList = comm.recv(source=MPI.ANY_SOURCE, tag=tags.DONE, status=status)
            resultsList = comm.recv(source=MPI.ANY_SOURCE, tag=tags.EXIT)
            print(resultsList)
            saveResults(resultsList, self.results_json_fname, self.results_csv_fname)
        else:
            # Worker processes execute code below
            name = MPI.Get_processor_name()
            manager_rank = rank % self.num_agents
            print("worker with rank %d on %s." % (rank, name))
            resultsList = []
            while True:
                comm.send(None, dest=manager_rank, tag=tags.READY)
                print(f'[W] rank: {rank} is ready.')
                task = comm.recv(source=manager_rank, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()
                if tag == tags.START:
                    print(f'[W] rank: {rank} starting task')
                    result = self.evaluate(self.problem, task, self.jobs_dir, self.results_dir)
                    elapsed_time = float(time.time() - start_time)
                    result['elapsed_time'] = elapsed_time
                    result['index'] = task['index']
                    print(result)
                    resultsList.append(result)
                    comm.send(result, dest=manager_rank, tag=tags.DONE)
                elif tag == tags.EXIT:
                    print(f'Exit rank={comm.rank}')
                    break
            comm.send(resultsList, dest=manager_rank, tag=tags.EXIT)

if __name__ == '__main__':
    args = PpoA3c.parse_args()
    search = PpoA3c(**vars(args))
    search.main()
