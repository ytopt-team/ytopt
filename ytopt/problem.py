import os

from pprint import pformat
from collections import OrderedDict

class Problem:
    """Problem specification for ytopt"""

    def __init__(self, cmd_frmt):
        self.__cmd_frmt = cmd_frmt
        self.__num_params = self.__cmd_frmt.count("{}")
        self.__space = [None for _ in range(self.num_params)]
        self.__def_values = [None for _ in range(self.num_params)]

    def __str__(self):
        return repr(self)

    def __repr__(self):
        rep_cmd_frmt = f'command format: "{self.cmd_frmt}"'
        starting_cmd = f'starting point: "{self.cmd_frmt.format(*self.__def_values)}"'
        rep_space = ""
        ln = len(str(self.__num_params))
        for i,v in enumerate(self.__space):
            tmp = '  {:'+str(ln)+'}: {} \n'
            rep_space += tmp.format(i, v)
        rep = f'{rep_cmd_frmt}\n{starting_cmd}\nspec of all dimensions:\n{rep_space} '
        return rep

    def spec_dim(self, p_id, p_space, default=None):
        """Add a dimension to the search space.
        Args:
            p_id (int): id of parameter.
            p_space (Object): space corresponding to the new dimension.
            default: default value of the new dimension, it must be compatible with the ``p_space`` given.
        """
        assert type(p_id) is int, f'p_id must be int type, got {type(p_id)} !'
        assert type(p_space) is tuple or type(p_space) is list, f'p_space must be tuple or list type, got {type(p_space)} !'
        self.__space[p_id] = p_space
        self.__def_values[p_id] = default

    @property
    def starting_point(self):
        """Starting point of the search space.
        """
        return self.__def_values

    @property
    def starting_point_asdict(self):
        return {k:v for k,v in enumerate(self.__def_values)}

    @property
    def space(self):
        return {k:v for k,v in enumerate(self.__space)}

    @property
    def params(self):
        return [i for i in range(self.__num_params)]

    @property
    def num_params(self):
        return self.__num_params

    @property
    def cmd_frmt(self):
        return self.__cmd_frmt

    def build_cmd(self, x):
        return self.__cmd_frmt.format(*x)

    def checkcfg(self):
        assert type(self.cmd_frmt) is str, f"cmd_frmt must be of type str when it is of type {type(self.cmd_frmt)}"
        assert sum(map(lambda x: x == None, self.space)) == 0, f"you still have unspecified parameters"
