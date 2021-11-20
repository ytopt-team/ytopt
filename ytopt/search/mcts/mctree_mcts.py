#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import itertools
import json
import pathlib
import tempfile
import heapq
import contextlib
import subprocess
import math
import datetime
import csv
import tool.invoke as invoke
from tool.support import *
from graphviz import Digraph
import time
from  datetime import timedelta
from datetime import datetime as DD

##### MCTS: Environment

import numpy as np 
import random,os,sys,time
from collections import namedtuple
from random import choice
from monte_carlo_tree_search_v1 import MCTS, Node
import matplotlib.pyplot as plt

random.seed(30)

N_repeat = 5

##### import mctree
import mctree_mcts_base as mctree
import tool.invoke as invoke
from multiprocessing import Process
import pickle
from statistics import median
from csv import reader, writer

# ##### import DT 
# import pandas as pd
# from sklearn import preprocessing
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor # Import Decision Tree Classifier
# from sklearn.model_selection import train_test_split # Import train_test_split function
# from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
# from sklearn import tree
# from sklearn.datasets import *
# from sklearn import tree
# from dtreeviz.trees import *
# from IPython.core.display import SVG
# import cairosvg
import collections 

##### MCTS Environment
_TTTB = namedtuple("Environment", "tup terminal depth parent ids trajectory")
        
class Environment(_TTTB, Node):
    
    outputs = [] 
    counter = 0
    visit = dict()
    thresh = [1]
    target_best = 1
    patient = 1

    def check_patient(self, n_patient):
        if len(set(np.array(env.outputs)[-n_patient:,0])) == 1:
            return True
        else:
            return False    
               
    def initialization(self):
        Environment.counter = 0
        Environment.visit = dict()
        Environment.thresh = [1]
        Environment.target_best = 1
        Environment.patient = 1       
    
    def find_children(state):
        if state.terminal:  # If the game is finished then no moves can be made
            return set()
        if state.tup.depth > max_depth_global:
            return set([Environment(None, True, state.tup.depth+1, state.tup, None, state.trajectory+(None,))])
        
        if state.ids == None:
            num_children = state.tup.get_num_children()
            if run_count == 0: ## only include restart 1:
                if state.depth < 2: 
                    init_id = collections.deque(maxlen=5)
                    init_id.append(random.sample(range(min(num_children,sys.maxsize)),min(num_children,1000))[0])
                    for _ in range(num_children):
                        item = state.tup.get_child(_)
                        if 'loop3) parallelize_thread' in item.nestexperiments[0].pragmalist[int(state.depth)-1]:
                            init_id.append(_)
                        if len(init_id) > 1:
                            break
            
            c_ids = random.sample(range(min(num_children,sys.maxsize)),min(num_children,1000))   
            random.shuffle(c_ids)
            state = state._replace(ids=tuple(c_ids))
                         
        results = []
        state.tup.derivatives = dict()
        ## non-exclusive
        if run_count == 0: ## make sure parallel is included at the beginning
            n_child = random.sample(state.ids, min(N_child,len(state.ids)))
            if state.depth < 2:
                for i_id in init_id:
                    if i_id not in n_child:
                        n_child.append(i_id)
                        random.shuffle(n_child)
        else:
            n_child   = random.sample(state.ids, min(N_child,len(state.ids)))
        for i in n_child:
            child = state.tup.get_child(i)
            child.depth = len(child.nestexperiments[0].pragmalist)+1
            results.append(Environment((child), False, child.depth, False, None, state.trajectory+(i,)))
        return set(results)
   
    def reward(state):        
        
        ## terminate state  
        if state.tup == None:
            experiment = state.parent
        else:
            experiment = state.tup
        if experiment.duration == None:
            ancestor = experiment.derived_from
            while ancestor.derived_from != None:
                ancestor = ancestor.derived_from
                
            mctree.run_experiment(d, experiment, ccargs=ccargs, execopts=execopts, 
                        writedot=False, 
                        dotfilter=None,
                        dotexpandfilter=None,
                        root=ancestor)
            
        res = experiment.duration
        if res == math.inf:
            worst = 30 #execopts.timeout 
            res = -0.1
            val = 0
            Target = np.mean(Environment.thresh[-N_window:])
            Environment.target_best = max(Environment.target_best, Target) 
            Environment.outputs.append([worst,experiment.depth-1,val, Environment.target_best])
            Environment.counter += 1  # count failure case
            Environment.visit[str(state.trajectory)] = state
            return  res
        elif str(state.trajectory) in Environment.visit.keys():
            val = ROOT_TIME/res.total_seconds()
            Target = np.mean(Environment.thresh[-N_window:])
            Environment.target_best = max(Environment.target_best, Target) 
            Environment.outputs.append([res.total_seconds(),10,val, Environment.target_best])
            Environment.thresh.append(val)
            if Environment.target_best <= val:
                return 0.5
            else:
                return 0
        else:
            val = ROOT_TIME/res.total_seconds()
            Target = np.mean(Environment.thresh[-N_window:])
            Environment.target_best = max(Environment.target_best, Target) 
            Environment.outputs.append([res.total_seconds(),experiment.depth-1,val, Environment.target_best])
            Environment.visit[str(state.trajectory)] = state
            
            ## win/lose reward 
            if ROOT_TIME != 1000:
                Environment.thresh.append(val)
            
            if ROOT_TIME == 1000:
                return 1
            
            if Environment.target_best <= val:
                return 1
            else:
                return 0

    def is_terminal(state):
        if state.tup == None:
            return True
        if state.tup.duration == math.inf:
            return True
        if state.tup.depth > max_depth_terminal: 
            return True
        else:
            return False
    
def save_model(file_path,save_tree):
    object_mcts = save_tree
    file_mcts = open(file_path, 'wb') 
    pickle.dump(object_mcts, file_mcts)    
    print ('.....save model.....',file_path)
    
def load_model(file_path):
    file_mcts = open(file_path, 'rb')  # open('filename_mcts.pkl', 'rb') 
    tree_loaded = pickle.load(file_mcts)
    file_mcts.close()
    for key in tree_loaded.children.keys():
        if key.parent == True:
            loaded_env = Environment(tup=(key.tup), terminal=False, depth=1, parent=True)
            break   
    print ('.....load model.....',file_path)
    return tree_loaded, loaded_env 

def update_depth(d,t,depth_list):
    if t < depth_list[d-1]:
        depth_list[d-1] = t
    return depth_list

def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 0: 
        return(True)  
    return(False)

if __name__ == "__main__":       
   
   if False:
       print ('void')
   else:
    ### take related arguments from mctree    
    args = mctree.main(argv=sys.argv)
    ccargs = mctree.parse_cc_cmdline(args.ccline)
    ccargs.polybench_time = args.polybench_time
    execopts = argparse.Namespace()
    
    execopts.ld_library_path = None
    if args.ld_library_path != None:
        execopts.ld_library_path = ':'.join(args.ld_library_path)
    execopts.timeout = 30
    if args.timeout != None:
        execopts.timeout = datetime.timedelta(seconds=args.timeout)
    execopts.polybench_time = args.polybench_time

    execopts.args = shcombine(arg=args.exec_arg,args=args.exec_args)

    outdir = mkpath(args.outdir)
    maxdepth = 5   ##############################################

    if args.keep:
        d = tempfile.mkdtemp(dir=outdir, prefix='mctree-')
    else:
        d = tempfile.mkdtemp(dir=outdir, prefix='mctree-')
    d = mkpath(d)     
          
    #### MCTS run commands     
    global max_depth_terminal
    global max_depth_global
    global ROOT_TIME
    global N_child
    global N_window
    global run_count
    N_patient = 10
    N_child  = int(10)
    N_window = int(10)
    max_depth = int(5) ## if 3, then 3 pragmas possible ==> mctree depth 4 
    max_depth_global = max_depth
    p_names = ['p0','p1','p2','p3','p4']
    r_idx  = 'poly_v1_test'
    w       = 0.1 # float(sys.argv[1])
    save_dir = 'w_'+str(w)+'_seed30_'+r_idx
    load_file = './tmp_w_0.1_xxx/model_xxx.pkl'

    w_2     = w # float(sys.argv[2])
    n_run   = 300 # int(sys.argv[3]) 170
    n_run_1 = 50 # int(sys.argv[4]) 0
    n_run_2 = n_run-n_run_1
    max_itr = 10000
    max_exp = 1001
    print ('\n.....RUN mcts.....')
    print('n_run:',n_run,'n_run_1:',n_run_1,'n_run_2:',n_run_2,'r_idx',r_idx)

    ############### start from beginning
    tree = MCTS(exploration_weight=w)
    root = mctree.extract_loopnests(d, ccargs=ccargs, execopts=execopts)
    env = Environment(tup=(root), terminal=False, depth=1, parent=True, ids=None, trajectory=('s',))   
    ###################################################    
    run_count = 0
    itr = 0
    while itr < max_itr:
        Patience = 0
        print ('=============================================================================================',run_count)
        max_depth_terminal_init = []
        for _ in range(n_run_1):
            max_depth_terminal_init.extend([i for i in range(1,max_depth+1)])
        max_depth_terminal_init = max_depth_terminal_init[:n_run_1]
        random.shuffle(max_depth_terminal_init)
        random.shuffle(max_depth_terminal_init)
        print ('exploration_weight init.......................................:',tree.exploration_weight)
        ###################### ROOT
        tree.start_time = time.time()
        print (max_depth_terminal_init)
        ROOT_TIME = 1000        # initial root time
        max_depth_terminal = 0
        tree._expand(env)
        reward = tree.evalaute([env])
        tree._backpropagate([env], reward)
        leaf = env
        cmdlines = 'cd '+str(leaf.tup.exppath)+' && '+str(leaf.tup.exppath)+'/base.exe'
        result = tree._dump_result(p_name=p_names,p_val=leaf.tup.nestexperiments[0].pragmalist, reward=reward, exppath=leaf.tup.exppath,time_v=env.tup.duration.total_seconds(),depth_v=str(0),cmd_v=str(cmdlines),init=True)
        tree.resultsList.append(result)
        tree.resultsList_current.append(result)
        ######################
        ROOT_TIME = env.tup.duration.total_seconds()
        print ('Root time',ROOT_TIME)  
        itr += 1
############################################################################ 
##################################################################  
        ## expand best sofar 
        for best_idx in range(len(tree.best_paths)):
            path   = tree.best_paths[best_idx][0]
            for best_item in range(len(path)-2,-1,-1):
                tree._expand(node=path[best_item], best=True,node_child=path[best_item+1])
###################################################################
#############################################################################         
        depth_list = [math.inf] * max_depth
        for _ in range(n_run_1):
            max_depth_terminal = max_depth_terminal_init.pop()
            print ('=========init==============',run_count,_, itr, 'terminal depth', max_depth_terminal)        
            path = tree.ask(env, True)
            reward = tree.evalaute(path)
            tree.tell(path, reward)
            print ('env.counter',env.counter)
            #print (env.target_best,env.patient)
            ###################################################################
            ## dump result
            leaf = path[-1]
            if leaf.tup == None:
                leaf = path[-2]
            try:    
                file1 = open(str(leaf.tup.exppath) + "/cc.txt","r")
                cmdlines = file1.readlines()[0][1:-2]
                file1.close()
            except FileNotFoundError:
                cmdlines=None        
            try: ### normal case 
                depth_list = update_depth(max_depth_terminal,leaf.tup.duration.total_seconds(),depth_list)
                result = tree._dump_result(p_name=p_names,p_val=leaf.tup.nestexperiments[0].pragmalist,reward=reward, exppath=leaf.tup.exppath,time_v=leaf.tup.duration.total_seconds(),depth_v=str(leaf.tup.depth-1),cmd_v=cmdlines,init=False)
                ## update best so far 
                tree.update_best(path,leaf.tup.duration.total_seconds())
                tree.success_paths.append(path)
                tree.searched_paths[tuple(path)] = ROOT_TIME / leaf.tup.duration.total_seconds()
            except AttributeError:
                inf_time = execopts.timeout # leaf.tup.duration
                result = tree._dump_result(p_name=p_names,p_val=leaf.tup.nestexperiments[0].pragmalist,reward=reward, exppath=leaf.tup.exppath,time_v=inf_time,depth_v=str(leaf.tup.depth-1),cmd_v=cmdlines,init=False)
                tree.fail_path_candi.append(path)
            tree.resultsList.append(result)
            tree.resultsList_current.append(result)
            itr += 1
            if itr > max_itr:
                break
            if leaf.tup.expnumber > max_exp:
                break
            print ('best_depth:',depth_list, np.argmin(depth_list[1:])+2) # np.argmin(depth_list[1:])+2)
        tree.dump_evals(save_dir,run_idx='_'+str(run_count))
        tree.write_graph(save_dir,n_run,env_output=env.outputs,run_idx='_'+str(run_count))
        tree.dump_evals(save_dir)
        tree.write_graph(save_dir,n_run,env_output=env.outputs)
        ######################################################################    
        ######################## save object 
        tree.visit = env.visit
        tree.searched.update(env.visit)
        tree.thresh = env.thresh   
        tree.target = env.target_best
        #save_model('./tmp_' + str(save_dir) + '/model_' + str(save_dir) + '_init_'+str(run_count)+'.pkl', tree)
        best_depth = np.argmin(depth_list[1:])+2
        tree.exploration_weight = w_2    
        print ('exploration_weight rollout.....................................:',tree.exploration_weight)  
############################################################################ 
##################################################################  

        ### update current failure cases
        for f_idx in range(len(tree.fail_path_candi)):
            f_item_0 = tree.fail_path_candi.pop()
            f_item = f_item_0[-1].tup.nestexperiments[0].pragmalist
            compare_result = [] 
            for b_idx in range(len(tree.success_paths)):
                b_item = tree.success_paths[b_idx][-1].tup.nestexperiments[0].pragmalist
                c_result = common_member(b_item, f_item)
                compare_result.append(c_result)
                if c_result == True:
                    break
            if not True in compare_result: ## all False
                tree.fail_paths.append(f_item_0)
                
        ### update previous failure cases
        for f_idx in range(len(tree.fail_paths)):
            f_item_0 = tree.fail_paths.pop()
            f_item = f_item_0[-1].tup.nestexperiments[0].pragmalist
            compare_result = [] 
            for b_idx in range(len(tree.success_paths)):
                b_item = tree.success_paths[b_idx][-1].tup.nestexperiments[0].pragmalist
                c_result = common_member(b_item, f_item)
                compare_result.append(c_result)
                if c_result == True:
                    break
            if not True in compare_result: ## all False
                tree.fail_paths.append(f_item_0)

        ## backprop worst sofar 
        for best_idx in range(len(tree.worst_paths)):
            path   = tree.worst_paths[best_idx][0]
            for best_item in range(len(path)-2,-1,-1):
                tree._expand(node=path[best_item], best=True,node_child=path[best_item+1])
            max_depth_terminal = path[-1].depth - 1
            reward = tree.evalaute(path)
            reward = -0.1
            tree.tell(path, reward)   
            ## dump result
            leaf = path[-1]
            if leaf.tup == None:
                leaf = path[-2]
            try:    
                file1 = open(str(leaf.tup.exppath) + "/cc.txt","r")
                cmdlines = file1.readlines()[0][1:-2]
                file1.close()
            except FileNotFoundError:
                cmdlines=None
            try:    
                result = tree._dump_result(p_name=p_names,p_val=leaf.tup.nestexperiments[0].pragmalist,reward=reward, exppath=leaf.tup.exppath,time_v=leaf.tup.duration.total_seconds(),depth_v=str(leaf.tup.depth-1),cmd_v=cmdlines,init=False)
                ## update best so far 
                tree.update_best(path,leaf.tup.duration.total_seconds())
            except AttributeError:
                inf_time = execopts.timeout # leaf.tup.duration            
                result = tree._dump_result(p_name=p_names,p_val=leaf.tup.nestexperiments[0].pragmalist,reward=reward, exppath=leaf.tup.exppath,time_v=inf_time,depth_v=str(leaf.tup.depth-1),cmd_v=cmdlines,init=False)
            tree.resultsList.append(result)
            tree.resultsList_current.append(result)
            itr += 1

        ## backprop fail sofar 
        for best_idx in range(len(tree.fail_paths)):
            path   = tree.fail_paths[best_idx]
            for best_item in range(len(path)-2,-1,-1):
                tree._expand(node=path[best_item], best=True,node_child=path[best_item+1])
            max_depth_terminal = path[-1].depth - 1
            #reward = tree.evalaute(path)
            reward = -0.1
            tree.tell(path, reward)   
            ## dump result
            leaf = path[-1]
            if leaf.tup == None:
                leaf = path[-2]
            try:    
                file1 = open(str(leaf.tup.exppath) + "/cc.txt","r")
                cmdlines = file1.readlines()[0][1:-2]
                file1.close()
            except FileNotFoundError:
                cmdlines=None
            try:    
                result = tree._dump_result(p_name=p_names,p_val=leaf.tup.nestexperiments[0].pragmalist,reward=reward, exppath=leaf.tup.exppath,time_v=leaf.tup.duration.total_seconds(),depth_v=str(leaf.tup.depth-1),cmd_v=cmdlines,init=False)
                ## update best so far 
                tree.update_best(path,leaf.tup.duration.total_seconds())
            except AttributeError:
                inf_time = execopts.timeout # leaf.tup.duration            
                result = tree._dump_result(p_name=p_names,p_val=leaf.tup.nestexperiments[0].pragmalist,reward=reward, exppath=leaf.tup.exppath,time_v=inf_time,depth_v=str(leaf.tup.depth-1),cmd_v=cmdlines,init=False)
            tree.resultsList.append(result)
            tree.resultsList_current.append(result)
            #itr += 1              

        ## backprop best sofar 
        for best_idx in range(len(tree.best_paths)):
            path   = tree.best_paths[best_idx][0]
            for best_item in range(len(path)-2,-1,-1):
                tree._expand(node=path[best_item], best=True,node_child=path[best_item+1])
            max_depth_terminal = path[-1].depth - 1
            reward = tree.evalaute(path)
            tree.tell(path, reward)   
            ## dump result
            leaf = path[-1]
            if leaf.tup == None:
                leaf = path[-2]
            try:    
                file1 = open(str(leaf.tup.exppath) + "/cc.txt","r")
                cmdlines = file1.readlines()[0][1:-2]
                file1.close()
            except FileNotFoundError:
                cmdlines=None
            try:    
                result = tree._dump_result(p_name=p_names,p_val=leaf.tup.nestexperiments[0].pragmalist,reward=reward, exppath=leaf.tup.exppath,time_v=leaf.tup.duration.total_seconds(),depth_v=str(leaf.tup.depth-1),cmd_v=cmdlines,init=False)
                ## update best so far 
                tree.update_best(path,leaf.tup.duration.total_seconds())
            except AttributeError:
                inf_time = execopts.timeout # leaf.tup.duration            
                result = tree._dump_result(p_name=p_names,p_val=leaf.tup.nestexperiments[0].pragmalist,reward=reward, exppath=leaf.tup.exppath,time_v=inf_time,depth_v=str(leaf.tup.depth-1),cmd_v=cmdlines,init=False)
            tree.resultsList.append(result)
            tree.resultsList_current.append(result)
            itr += 1            
            
###################################################################
#############################################################################          
        
        for _ in range(n_run_2):
            max_depth_terminal = int(best_depth) ##max_depth ## max_depth_global
            print ('===========================',run_count,_+n_run_1, itr, 'terminal depth', max_depth_terminal)
            path = tree.ask(env, False)
            #print('path:')
            #print(path)
            reward = tree.evalaute(path)
            tree.tell(path, reward)
            print ('env.counter',env.counter)
            ###################################################################
            ## dump result
            leaf = path[-1]
            if leaf.tup == None:
                leaf = path[-2]
            try:    
                file1 = open(str(leaf.tup.exppath) + "/cc.txt","r")
                cmdlines = file1.readlines()[0][1:-2]
                file1.close()
            except FileNotFoundError:
                cmdlines=None
            try:    
                result = tree._dump_result(p_name=p_names,p_val=leaf.tup.nestexperiments[0].pragmalist,reward=reward, exppath=leaf.tup.exppath,time_v=leaf.tup.duration.total_seconds(),depth_v=str(leaf.tup.depth-1),cmd_v=cmdlines,init=False)
                ## update best so far 
                tree.update_best(path,leaf.tup.duration.total_seconds())
                tree.success_paths.append(path)
                tree.searched_paths[tuple(path)] = ROOT_TIME / leaf.tup.duration.total_seconds()
            except AttributeError:
                inf_time = execopts.timeout # leaf.tup.duration            
                result = tree._dump_result(p_name=p_names,p_val=leaf.tup.nestexperiments[0].pragmalist,reward=reward, exppath=leaf.tup.exppath,time_v=inf_time,depth_v=str(leaf.tup.depth-1),cmd_v=cmdlines,init=False)
                tree.fail_path_candi.append(path)
            tree.resultsList.append(result)
            tree.resultsList_current.append(result)
            itr += 1
            ###################################################################     
            if (_+1) % 50 == 0:
                tree.dump_evals(save_dir,run_idx='_'+str(run_count))
                #tree.write_graph(save_dir,d_name,n_run,env_output=env.outputs,run_idx='_'+str(run_count))
                tree.dump_evals(save_dir)
                tree.write_graph(save_dir,n_run,env_output=env.outputs)

            # patient for the same obj val  
            patient_tmp = env.check_patient(N_patient)          
            if patient_tmp:
                break
            # patient for the target improvement
            if env.target_best == env.patient:
                Patience += 1
                if Patience > 50:
                    break
            else:
                Patience = 0
                env.patient = env.target_best
            if itr > max_itr:
                break 
            if leaf.tup.expnumber > max_exp:
                break
        feature_col = [] 
        for ___ in range(best_depth):
            feature_col.append('p'+str(___))
        
        tree.dump_evals(save_dir,run_idx='_'+str(run_count))
        tree.write_graph(save_dir,n_run,env_output=env.outputs,run_idx='_'+str(run_count)) 
        tree.dump_evals(save_dir)
        tree.write_graph(save_dir,n_run,env_output=env.outputs)       
        #tree.run_dt(save_dir=save_dir, run_idx='_'+str(run_count) ,feature_cols_=feature_col,root_time=ROOT_TIME,dt_depth=3)
        #tree.run_dt(save_dir=save_dir, run_idx='_all',feature_cols_=feature_col,root_time=ROOT_TIME,dt_depth=3)
        ## find/update best&worst config 
        alpha = 0.95
        tree.best_alpha = [] 
        tree.worst_alpha = []
        sorted_keys = sorted(tree.searched_paths, key=tree.searched_paths.get) 
        for s_key in sorted_keys[int(len(tree.searched_paths)*alpha):]:
            tree.best_alpha.append([list(s_key),tree.searched_paths[s_key]])
        tree.best_paths = tree.best_alpha
        for s_key in sorted_keys:
            w_item_0 = list(s_key)
            w_item = w_item_0[-1].tup.nestexperiments[0].pragmalist
            compare_result = [] 
            for b_idx in range(len(tree.best_alpha)):
                b_item = tree.best_alpha[b_idx][0][-1].tup.nestexperiments[0].pragmalist
                compare_result.append(common_member(b_item, w_item))
            if not True in compare_result: ## all False
                tree.worst_alpha.append([s_key,tree.searched_paths[s_key]])        
            if len(tree.worst_alpha) > int(len(tree.searched_paths)*(1-alpha)):
                break 
        tree.worst_paths = tree.worst_alpha
            
        ### update current failure cases
        for f_idx in range(len(tree.fail_path_candi)):
            f_item_0 = tree.fail_path_candi.pop()
            f_item = f_item_0[-1].tup.nestexperiments[0].pragmalist
            compare_result = [] 
            for b_idx in range(len(tree.success_paths)):
                b_item = tree.success_paths[b_idx][-1].tup.nestexperiments[0].pragmalist
                c_result = common_member(b_item, f_item)
                compare_result.append(c_result)
                if c_result == True:
                    break
            if not True in compare_result: ## all False
                tree.fail_paths.append(f_item_0)
                
        ### update previous failure cases
        for f_idx in range(len(tree.fail_paths)):
            f_item_0 = tree.fail_paths.pop()
            f_item = f_item_0[-1].tup.nestexperiments[0].pragmalist
            compare_result = [] 
            for b_idx in range(len(tree.success_paths)):
                b_item = tree.success_paths[b_idx][-1].tup.nestexperiments[0].pragmalist
                c_result = common_member(b_item, f_item)
                compare_result.append(c_result)
                if c_result == True:
                    break
            if not True in compare_result: ## all False
                tree.fail_paths.append(f_item_0)
                
        print ('len(tree.best_paths)',len(tree.best_paths))
        print ('len(tree.worst_paths)',len(tree.worst_paths))
        print ('len(tree.fail_paths)',len(tree.fail_paths))
        print ('Failure cases....',env.counter, '....out of', n_run)
        print ('Mean time....',np.mean(env.outputs,axis=0),'std time....',np.std(env.outputs,axis=0))
        
        #######################################################################    
        ######################## save object 
        tree.visit = env.visit
        tree.searched.update(env.visit)
        tree.thresh = env.thresh
        tree.target = env.target_best     
        
        ## check to finish 
        if itr > max_itr:
            break 
        if leaf.tup.expnumber > max_exp:
            break            
        ## restart
        tree.initialization()    
        env.initialization()
        env.patient = 1
        run_count += 1
