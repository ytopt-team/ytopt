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
import dl4lb

seed = 12345

m, n = 32, 8

nparam, nb_classes = 32, 8

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def create_parser():
    'command line parser for keras'

    parser = argparse.ArgumentParser(add_help=True)
    group = parser.add_argument_group('required arguments')

    for id in range(nparam):
        parser.add_argument('--p%d'%id, action='store', dest='p%d'%id,
                            nargs='?', const=2, type=str, default='0',
                            help='parameter p%d value'%id)

    return(parser)

parser = create_parser()
cmdline_args = parser.parse_args()
param_dict = vars(cmdline_args)

one_hot = get_one_hot(np.array(list(range(nb_classes))), nb_classes)
#print(one_hot)

decode_dict = {}
for i in range(nb_classes):
    decode_dict[str(i)] = one_hot[i]


vals = []
for id in range(nparam):
    val = param_dict['p%d'%id]
    vals.append(decode_dict[val])

x=np.array(vals)
print(x)

def loss_function(x):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    L,S,A,R,Y = dl4lb.__generate_test_config(m,n)
    l = dl4lb.__test_loss(L,S,A,R,x)
    return l

pval = loss_function(vals)
print('OUTPUT:%1.3f'%pval)
