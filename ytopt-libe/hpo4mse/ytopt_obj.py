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
    H_o['Object'] = y
    H_o['elapsed_sec'] = time.time() - start_time
   # output the best
    bestfile = '/tmp/best.txt'
    best = float(1000)
    if os.path.isfile(bestfile) == False:
        f = open(bestfile, "w")
        f.write(format(y, '.12f'))
        f.close()
    else:
        f = open(bestfile, "r")
        best = float(f.read())
    if best >= y:
        best = y
        f = open(bestfile, "w")
        f.write(format(best, '.12f'))

    H_o['Best'] = best

    return H_o, persis_info


def myobj(point: dict, params: list, workerID: int):
    def plopper_func(x, params):
        obj = Plopper('./kan.py', './')
        x = np.asarray_chkfinite(x)
        value = [point[param] for param in params]
        #print(value)
        #os.system("./processexe.pl exe.pl " +str(value[0]))
        #os.environ["OMP_NUM_THREADS"] = str(value[0])
        params = [i.upper() for i in params]
        result = obj.findRuntime(value, params, workerID)
        return result

    x = np.array([point[f'p{i}'] for i in range(len(point))])
    results = plopper_func(x, params)
    # print('CONFIG and OUTPUT', [point, results], flush=True)
    return results
