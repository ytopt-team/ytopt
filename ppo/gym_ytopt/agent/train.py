#!/usr/bin/env python3

import tensorflow as tf
import json
import math

from mpi4py import MPI
from ppo.baselines.common import set_global_seeds
from ppo.baselines import bench
import os.path as osp
from ppo.baselines import logger

from ppo.gym_ytopt.envs import YtoptEnvParallel
from ppo.gym_ytopt.agent import  pposgd_simple, lstm_policy, mlp_policy
import ppo.baselines.common.tf_util as U
from math import inf


class Train:
    """Training class for ppo. Sequential evaluation of workers.
    Args:
        problem (Problem): a problem from one of the benchmarks
        policy_fn (func): return a policy instance
        num_episodes_per_batch (int): number of episodes per batch of update (SYNC)
        num_episodes (int): total number of episodes to sample
        seed (int): seed of the current agent
    """

    def __init__(self, problem, rank_workers, policy_fn,
        num_episodes=math.inf,
        seed=2018,
        comm=None,
        tags=None,
        max_time=inf):
        self.rank_workers = rank_workers
        self.problem = problem
        self.seed = seed
        self.policy_fn = policy_fn
        self.num_episodes_per_batch = len(rank_workers)
        self.num_episodes = num_episodes
        self.max_time = max_time
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.tags = tags

    def train(self):
        num_episodes = self.num_episodes
        seed = self.seed
        policy_fn = self.policy_fn

        rank = self.comm.Get_rank()
        sess = U.single_threaded_session()
        sess.__enter__()
        if rank == 0:
            logger.configure()
        else:
            logger.configure(format_strs=[])
        workerseed = seed + 10000 * self.comm.Get_rank() if seed is not None else None
        set_global_seeds(workerseed)

        # MAKE ENV_NAS
        # num_episodes = 1000
        episode_length = len(self.problem.space.keys())
        timesteps_per_actorbatch = episode_length * self.num_episodes_per_batch
        # num_timesteps = timesteps_per_actorbatch * num_episodes

        env = YtoptEnvParallel(rank_workers=self.rank_workers, problem=self.problem)

        print(f'[A, r={rank}] learning')
        pposgd_simple.learn(env, policy_fn,
            # max_timesteps=int(num_timesteps),
            max_seconds=self.max_time,
            timesteps_per_actorbatch=timesteps_per_actorbatch,
            clip_param=0.2,
            entcoeff=0.01,
            optim_epochs=4,
            optim_stepsize=1e-3,
            optim_batchsize=15,
            gamma=0.99,
            lam=0.95,
            schedule='constant',
            comm=self.comm,
            tags=self.tags,
            rank_workers=self.rank_workers

        )
        env.close()
