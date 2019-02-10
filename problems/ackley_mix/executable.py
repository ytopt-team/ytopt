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
    'command line parser'
    
    parser = argparse.ArgumentParser(add_help=True)
    group = parser.add_argument_group('required arguments')
    parser.add_argument('--p0', action='store', dest='p0',
                        nargs='?', const=2, type=float, default=-15.0,
                        help='parameter p0 value')
    parser.add_argument('--p1', action='store', dest='p1',
                        nargs='?', const=2, type=float, default=-15.0,
                        help='parameter p1 value')
    parser.add_argument('--p2', action='store', dest='p2',
                        nargs='?', const=2, type=int, default=-15,
                        help='parameter p2 value')
    parser.add_argument('--p3', action='store', dest='p3',
                        nargs='?', const=2, type=int, default=-15,
                        help='parameter p3 value')
    parser.add_argument('--p4', action='store', dest='p4',
                        nargs='?', const=2, type=str, default='-15',
                        help='parameter p4 value')
    parser.add_argument('--p5', action='store', dest='p5',
                        nargs='?', const=2, type=str, default='-15',
                        help='parameter p5 value')

    return(parser)

parser = create_parser()
cmdline_args = parser.parse_args()
param_dict = vars(cmdline_args)
print(param_dict)
p0 = param_dict['p0']
p1 = param_dict['p1']
p2 = param_dict['p2']
p3 = param_dict['p3']
p4 = int(param_dict['p4'])
p5 = int(param_dict['p5'])


x=np.array([p0,p1,p2,p3,p4,p5])

def ackley( x, a=20, b=0.2, c=2*pi ):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = sum( x**2 )
    s2 = sum(cos( c * x ))
    return -a*exp( -b*sqrt( s1 / n )) - exp( s2 / n ) + a + exp(1)

pval = ackley(x, a=20, b=0.2, c=2*pi)
print('OUTPUT:%1.3f'%pval)
