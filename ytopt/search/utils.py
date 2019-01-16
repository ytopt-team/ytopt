import csv
import json
import math
import os
import errno
import re
import subprocess
import sys
import time
from importlib import import_module
from string import Template


def saveResults(resultsList, json_fname, csv_fname):
    print(resultsList)
    print(json.dumps(resultsList, indent=4, sort_keys=True))
    with open(json_fname, 'w') as outfile:
        json.dump(resultsList, outfile, indent=4, sort_keys=True)

    keys = resultsList[0].keys()
    with open(csv_fname, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(resultsList)

def saveMetaData(metaDataDict, json_fname):
    with open(json_fname, 'w') as outfile:
        json.dump(metaDataDict, outfile, indent=4, sort_keys=True)



def load_attr_from(str_full_module):
    """
        Args:
            - str_full_module: (str) correspond to {module_name}.{attr}
        Return: the loaded attribute from a module.
    """
    if type(str_full_module) == str:
        split_full = str_full_module.split('.')
        str_module = '.'.join(split_full[:-1])
        str_attr = split_full[-1]
        module = import_module(str_module)
        return getattr(module, str_attr)
    else:
        return str_full_module

def load_from_file(fname, attribute):
    dirname, basename = os.path.split(fname)
    sys.path.insert(0, dirname)
    module_name = os.path.splitext(basename)[0]
    module = import_module(module_name)
    return getattr(module, attribute)

def generic_loader(target, attribute):
    '''Load attribute from target module
    Args:
        - target: either path to python file, or dotted Python package name
        - attribute: name of the attribute to load from the target module
    '''
    # assert attribute in ['Problem', 'run']
    if not isinstance(target, str):
        return target
    if os.path.isfile(os.path.abspath(target)):
        target_file = os.path.abspath(target)
        return load_from_file(target_file, attribute)
    else:
        return load_attr_from(target)

def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define MPI message tags
tags = enum('READY', 'DONE', 'EXIT', 'START')


def setup_expfolder(exp_dir, job_dir, res_dir):
    if exp_dir[-1] == '/':
        exp_dir = exp_dir[:-1]
    if not os.path.exists(exp_dir):
        try:
            os.makedirs(exp_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        print('Created experiments directory at: ', os.path.abspath(exp_dir))
    if job_dir[-1] == '/':
        job_dir = job_dir[:-1]
    if not os.path.exists(job_dir):
        try:
            os.makedirs(job_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        print('Created jobs directory at: ', os.path.abspath(job_dir))
    if res_dir[-1] == '/':
        res_dir = res_dir[:-1]
    if not os.path.exists(res_dir):
        try:
            os.makedirs(res_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        print('Created results directory at: ', os.path.abspath(res_dir))