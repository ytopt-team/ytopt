import gym
import numpy as np
from gym import spaces


class YtoptEnv(gym.Env):

    eval_counter = 0

    def __init__(self, evaluate, problem):

        self.evaluate = evaluate
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
        result = self.evaluate(conv_action, YtoptEnv.eval_counter)
        YtoptEnv.eval_counter += 1

        # ob, reward, terminal
        # cost is minimization when ppo is maximization
        return self._state, -result['cost'], terminal, {}

    def reset(self):
        self.__init__(self.evaluate, self.problem)
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
