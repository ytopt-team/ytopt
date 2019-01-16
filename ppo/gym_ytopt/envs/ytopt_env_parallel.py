import gym
import numpy as np
from gym import spaces
from mpi4py import MPI
import time

def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define MPI message tags
tags = enum('READY', 'DONE', 'EXIT', 'START')

comm = MPI.COMM_WORLD   # get MPI communicator object
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

class YtoptEnvParallel(gym.Env):

    eval_counter = 0
    start_time = time.time()

    def __init__(self, rank_workers, problem):

        self.rank_workers = rank_workers
        self.problem = problem
        self.observation_space = spaces.Box(low=-0, high=self.max_action_size, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_action_size)

        self._state = np.array([1.])
        self.action_buffer = []
        self.num_timesteps = len(problem.params)

    def step(self, action, index):

        self.action_buffer.append(action)

        if len(self.action_buffer) < self.num_timesteps:
            terminal = False
            reward = 0
            self._state = np.array([float(action)])
            return self._state, reward, terminal, {}

        conv_action = self.index2tokens(self.action_buffer)

        terminal = True
        self.action_buffer = []
        self._state = np.array([1.])

        # EXECUTION
        data = comm.recv(source=MPI.ANY_SOURCE, tag=tags.READY, status=status)
        source = status.Get_source()
        print(f'[ENV] rank: {rank} found available worker -> rank={source}')
        task = {}
        task['x'] = conv_action
        task['index'] = index
        task['eval_counter'] = YtoptEnvParallel.eval_counter
        task['rank_master'] = rank
        elapsed_time = float(time.time() - YtoptEnvParallel.start_time)
        task['start_time'] = elapsed_time
        # print('Sending task {} to worker {}'.format (eval_counter, source))
        print(f'[ENV] rank: {rank} send task to rank={source}')
        comm.send(task, dest=source, tag=tags.START)
        YtoptEnvParallel.eval_counter += 1

        # ob, reward, terminal
        # cost is minimization when ppo is maximization
        return self._state, 0, terminal, {}

    def get_reward_ready(self):
        data = comm.recv(source=MPI.ANY_SOURCE, tag=tags.DONE, status=status)
        return data, -data['cost']

    def reset(self):
        self.__init__(self.rank_workers, self.problem)
        return self._state

    def render(self, mode='human', close=False):
        pass

    @property
    def max_action_size(self):
        space = self.problem.space
        mx = 0
        for k in space:
            mx = max(mx, len(space[k]))
        return mx

    def index2tokens(self, index_list):
        token_list = []
        space = self.problem.space
        for i, k in enumerate(space.keys()):
            index = index_list[i]
            f_index = index / self.max_action_size # float index
            n_index = int(f_index * len(space[k]) + 0.5) # new index
            token = space[k][n_index] # assuming that space[k] is a list
            token_list.append(token)
        return token_list
