"""
This module is a wrapper around an example ytopt objective function
"""
__all__ = ['init_obj']

import numpy as np
import os
import time
from plopper import Plopper

start_time = time.time()

def init_obj(H, persis_info, sim_specs, libE_info):
    point = {}
    for field in sim_specs['in']:
        point[field] = np.squeeze(H[field])

    y = myobj(point, sim_specs['in'], libE_info['workerID'])  # ytopt objective wants a dict
    H_o = np.zeros(2, dtype=sim_specs['out'])
    H_o['objective'] = y
    H_o['elapsed_sec'] = time.time() - start_time

    return H_o, persis_info

def myobj(point: dict, params: list, workerID: int):
    def plopper_func(x, params):
        obj = Plopper('./dlp.py', './')
        x = np.asarray_chkfinite(x)
        value = [point[param] for param in params]
        #print(value)
        params = [i.upper() for i in params]
        #result = float(obj.findRuntime(value, params, workerID))
        result = obj.findRuntime(value, params, workerID)
        return result

    x = np.array([point[f'p{i}'] for i in range(len(point))])
    results = plopper_func(x, params)
    # print('CONFIG and OUTPUT', [point, results], flush=True)
    return results
