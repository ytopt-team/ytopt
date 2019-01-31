#!/usr/bin/env python
from __future__ import print_function
import re
import os
import sys
import time
import json
import math
import os
import argparse
import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
seed = 12345

def create_parser():
    'command line parser for keras'

    parser = argparse.ArgumentParser(add_help=True)
    group = parser.add_argument_group('required arguments')

    for id in range(1,11):
        parser.add_argument('--p%d'%id, action='store', dest='p%d'%id,
                            nargs='?', const=2, type=int, default=-4,
                            help='parameter p%d value'%id)

    return(parser)

parser = create_parser()
cmdline_args = parser.parse_args()
param_dict = vars(cmdline_args)

p1 = param_dict['p1']
p2 = param_dict['p2']
p3 = param_dict['p3']
p4 = param_dict['p4']
p5 = param_dict['p5']
p6 = param_dict['p6']
p7 = param_dict['p7']
p8 = param_dict['p8']
p9 = param_dict['p9']
p10 = param_dict['p10']

x=np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10])

def powell( x ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    n4 = ((n + 3) // 4) * 4
    if n < n4:
        x = np.append( x, np.zeros( n4 - n ))
    x = x.reshape(( 4, -1 ))  # 4 rows: x[4i-3] [4i-2] [4i-1] [4i]
    f = np.empty_like( x )
    f[0] = x[0] + 10 * x[1]
    f[1] = sqrt(5) * (x[2] - x[3])
    f[2] = (x[1] - 2 * x[2]) **2
    f[3] = sqrt(10) * (x[0] - x[3]) **2
    return sum( f**2 )

pval = powell( x )
print('OUTPUT:%1.3f'%pval)






