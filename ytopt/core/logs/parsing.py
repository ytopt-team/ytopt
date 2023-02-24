import argparse
import datetime
import json
import os
import sys
from shutil import copyfile



HERE = os.path.dirname(os.path.abspath(__file__))
now = '_'.join(str(datetime.datetime.now(
    datetime.timezone.utc)).split(":")[0].split(" "))


def parseline_json(line, data):
    line = "".join(line)
    date = line.split('|')[0]
    jsn_str = line.split('>>>')[-1]
    info = json.loads(jsn_str)
    if data.get(info['type']) == None:
        data[info['type']] = list()
    value = info['type']
    info['timestamp'] = date[:10] + ' ' + date[10:]
    info.pop('type')
    data[value].append(info)


def parseline_reward(line, data):
    data['raw_rewards'].append(float(line[-1]))


def parseline_arch_seq(line, data):
    i_sta = line.index("'arch_seq':") + 1
    i_end = i_sta
    while not ']' in line[i_end]:
        i_end += 1
    l = []
    for i in range(i_sta, i_end+1):
        l.append(float(line[i].replace('[', '').replace(
            ',', '').replace(']', '').replace('}', '')))
    data['arch_seq'].append(l)


def parsing(f, data):
    line = f.readline()
    while line:
        line = line.split()
        if "y:" in line:
            parseline_reward(line, data)
            parseline_arch_seq(line, data)
        elif ">>>" in line:
            parseline_json(line, data)

        line = f.readline()


def add_subparser(subparsers):
    subparser_name = 'parse'
    function_to_call = main

    parser_parse = subparsers.add_parser(
        subparser_name, help='Tool to parse "ytopt.log" and produce a JSON file.')
    parser_parse.add_argument('path', type=str, help=f'The parsing script takes only 1 argument: the relative path to the log file.')

    return subparser_name, function_to_call


def main(path, *args, **kwargs):
    print(f'Path to ytopt.log file: {path}')

    data = dict()
    if len(path.split('/')) >= 3:
        data['fig'] = path.split('/')[-3] + '_' + now
        workload_in_path = True
    else:
        workload_in_path = False
        data['fig'] = 'data_' + now

    data['raw_rewards'] = list()
    data['arch_seq'] = list()

    with open(path, 'r') as flog:
        print('File has been opened')
        parsing(flog, data)
    print('File closed')

    with open(data['fig']+'.json', 'w') as fjson:
        print(f'Create json file: {data["fig"]+".json"}')
        json.dump(data, fjson, indent=2)
    print('Json dumped!')

    print(f'len raw_rewards: {len(data["raw_rewards"])}')
    print(f'len arch_seq   : {len(data["arch_seq"])}')
