"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import sys
import numpy as np 
import random
from random import choice
import time, csv, os, pickle
from graphviz import Digraph

random.seed(20)

##### import DT 
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *
from IPython.core.display import SVG
import cairosvg 
import collections 

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(float)  # total reward of each node 
        self.N = defaultdict(int)  # total visit count for each node
        self.O = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.resultsList = [] 
        self.resultsList_current = [] 
        self.start_time = None
        self.unexp = 0.5
        self.visit = dict()
        self.thresh = []
        self.target = 1.0
        self.searched = dict()
        self.best_path = None
        self.best_sofar = math.inf
        self.best_paths = []
        self.worst_path = None
        self.worst_sofar = 0
        self.worst_path_candi = collections.deque(maxlen=100)
        self.worst_paths = []
        ## fail/success
        self.fail_path_candi = collections.deque(maxlen=100)
        self.fail_paths = []
        self.success_paths = []
        self.searched_paths = dict()
        self.best_alpha = []
        self.worst_alpha = []
        
    def initialization(self):
        self.Q = defaultdict(float)  # total reward of each node 
        self.N = defaultdict(int)  # total visit count for each node
        self.O = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node 
#         self.exploration_weight = exploration_weight
#         self.resultsList = []
#         self.resultsList_current = []  
        self.start_time = None
#         self.unexp = 0.5
        self.visit = dict()
        self.thresh = []
        self.target = 1.0
        self.best_path = None
        self.best_sofar = math.inf
        self.worst_path = None
        self.worst_sofar = 0
        self.worst_path_candi = collections.deque(maxlen=100)
        self.fail_path_candi = collections.deque(maxlen=100)
#         self.searched = dict()      
        
    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def ask(self, node, init = False):
        "Make the tree one layer better. (Train for one iteration.)"
        path = []
        path.extend([node]) 
        self._expand(node)
        while True:
            p = self._select(node,init)
            leaf = p[-1]
            self._expand(leaf)
            node = leaf
            path.extend(p[1:])
            if node.is_terminal():
                break
        for i in path:
            self.O[i] += 1 ##### JK this is for debuging purpose
        return path
    
    def evalaute(self, path):
        leaf = path[-1]
        reward = self._simulate(leaf)
        return reward

    def tell(self, path, reward):
        #reward = self._simulate(leaf)
        self._backpropagate(path, reward)
        
    def _select(self, node, init):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node.is_terminal() or node not in self.children or not self.children[node]:
                return path
            else:
                if init:
                    unexplored = self.children[node] - self.children.keys()
                    if unexplored:       
                        n = choice(list(unexplored))
                        path.append(n)
                        return path                    
                if random.uniform(0,1) < self.unexp:
                    unexplored = self.children[node] - self.children.keys()
                    if unexplored:       
                        n = choice(list(unexplored))
                        path.append(n)
                        return path
                node = self._uct_select(node, init)
                    
    def _expand(self, node, best=False, node_child=None):
        "Update the `children` dict with the children of `node`"
        if best == True:
            if node in self.children: ## already expanded
                if not node_child in self.children[node]: # if node_child not in children
                    self.children[node].add(node_child)
                    return
                return
            else: # not expanded, then find children 
                self.children[node] = node.find_children() 
                if not node_child in self.children[node]: # if node_child not in children
                    self.children[node].add(node_child)
                    return      
                return 
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()
        
    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        while True:
            if node.is_terminal(): 
                reward = node.reward()
                return reward  
            node = node.find_random_child() 
    
    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        
        for node in reversed(path):
            self.O[node] -= 1
            self.N[node] += 1
            self.Q[node] += reward   

    def _uct_select(self, node, init):
        "Select a child of node, balancing exploration & exploitation"
#         assert all(n in self.children for n in self.children[node])
        log_N_vertex = math.log(1.+self.N[node])
        def uct(n):
            "Upper confidence bound for trees"  
            return self.Q[n]/(1.+self.N[n]) + self.exploration_weight * math.sqrt(2*log_N_vertex/(1.+self.N[n]))
        if init:
            return choice(list(self.children[node]))
        else:
            return max(self.children[node], key=uct)     

    def dump_evals(self,save_dir=None,run_idx='_all'):
        dir_path  = './tmp_' + str(save_dir) + '/'
        file_name = 'results_' + str(save_dir) +str(run_idx) +'.csv'
        if not os.path.exists(dir_path):
            os.mkdir(dir_path) 
        with open(dir_path+file_name, 'w') as fp:
            if run_idx=='_all':   
                columns = self.resultsList[0].keys()
                writer = csv.DictWriter(fp, columns)
                writer.writeheader()
                writer.writerows(self.resultsList)
            else:
                columns = self.resultsList_current[0].keys()
                writer = csv.DictWriter(fp, columns)
                writer.writeheader()
                writer.writerows(self.resultsList_current)
        print ('.....dump evaluations.....',dir_path+file_name)    
            
    def _dump_result(self,p_name,p_val,reward,exppath=None,time_v=None,depth_v=None,cmd_v=None,init=False):
        result  = {}
        if init:
            for p in p_name:
                result[p] = None
        for p,val in zip(p_name,p_val):
            result[p] = val
        result['objective'] = reward 
        result['elapsed_sec'] = time.time() - self.start_time
        if exppath:
            result['exppath'] = str(exppath)
        if time_v:
            result['time'] = time_v
        if depth_v:
            result['depth'] = depth_v
        if cmd_v:
            result['cmdline'] = cmd_v
        return result        
        
    def find_best(self,re=False,start=0):
        best_sofar = math.inf
        best_sofar_id = None
        for i in range(start,len(self.resultsList)):
            item = self.resultsList[i]
#             val  = item['objective']
            val  = item['time']
            if val < best_sofar:
                best_sofar = val
                best_sofar_id = item
                best_sofar_depth = item['depth']
        print ('best_sofar_item',best_sofar_id, 'best_sofar_time', best_sofar)
        if re:
            return best_sofar, best_sofar_depth

    def update_best(self,path,time):
        if not time in np.array(self.best_paths):
            if time < self.best_sofar:
                self.best_sofar = time
                self.best_path = path

        if not time in np.array(self.worst_paths):
            if time > self.worst_sofar:
                self.worst_sofar = time
                self.worst_path = path
                self.worst_path_candi.append([self.worst_path,self.worst_sofar])
                 
                
    def write_graph(self,save_dir,n_run,env_output=None,mctree=False,run_idx='_all'):
        dir_path  = './tmp_' + str(save_dir) + '/'
        file_name = 'graph_' + str(save_dir) + str(run_idx)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        if env_output:
            np.save(dir_path+file_name+'.npy',env_output)
            print ('.....write ouputs.....',dir_path+file_name+'.npy')
#         if mctree == True:
#             dot = Digraph(comment='MCTS Tree')
#             for key in self.children.keys(): 
#                 if key.tup != None:
#                     node_idx  = str(id(key.tup))
#                     node_desc = str(key.tup)
#                     if key.tup.duration == math.inf:
#                         dot.attr('node', style='filled', color='lightgrey')
#                         dot.node(node_idx, node_desc)   
#                     else:
#                         dot.attr('node', style='', color='black')
#                         dot.node(node_idx, node_desc)
#                     if key.tup.derived_from != None:
#                         node_edge_from_idx = str(id(key.tup.derived_from))
#                         dot.edge(node_edge_from_idx,node_idx)        
#             dot.render(dir_path+file_name+'.gv', view=True)
#             print ('.....write graph.....',dir_path+file_name+'.gv.pdf')           

    def run_dt(self,save_dir,run_idx,feature_cols_,root_time,dt_depth):
        # read data 
        dir_path  = './tmp_' + str(save_dir) + '/'
        file_name = 'results_' + str(save_dir) +str(run_idx) +'.csv'
        save_name = 'results_' + str(save_dir) +str(run_idx) +'.svg'  

        # col_names = ['p0', 'p1', 'p2', 'objective','label','time','speedup']
        pima = pd.read_csv(dir_path+file_name, header=0) # , names=col_names)
        ### preprocess data 
        feature_cols = feature_cols_
        pima = pima[feature_cols+['time']]
        pima = pima.drop_duplicates()   ##### drop duplicates
        pima = pima.dropna()            ##### drop missing 
        y = root_time / pima.time # Target variable
        X_0 = pima[feature_cols] # Features
        X = X_0.copy()
        le = preprocessing.OneHotEncoder()
        le.fit(X_0,y)
        X = le.transform(X_0)
        # fit DT 
        X_train, y_train = X, y
        # Create Decision Tree object
        clf = DecisionTreeRegressor(criterion="mse", max_depth=dt_depth)
        # Train Decision Tree
        clf = clf.fit(X_train,y_train)
        # generate tree
        feature_name = []
        for i in range(len(feature_cols)):
            mystring = 'D'+str(i+1)+'_'
            feature_name.extend(  [mystring+s for s in list(le.categories_[i])])
        # X_new = X.toarray()[1]  # random sample from training
        viz = dtreeviz(clf,
                       X.toarray(),
                       y,
                       target_name='speedup',
                       feature_names=feature_name)
        #                X=X_new, 
        #               show_just_path=True,
        #               fancy=False)
        viz.save(dir_path+save_name)
        cairosvg.svg2png(url=dir_path+save_name, write_to=dir_path+save_name+'.png',scale=2.0)
        # find leaf 
        # select pragma to include
#         return pragma_list           
            
class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """    
        
    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
    