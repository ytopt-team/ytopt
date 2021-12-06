#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv
import pathlib


def makeplot(filename):
    indices=[]
    idxtime=[]

    with open(filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            idx = int(row[0])
            time = float(row[1])
            indices.append(idx)
            idxtime.append(time)

    idxlen = max(indices)+1

    bestidx = [indices[0]]
    besttime = [idxtime[0]]
    for p in zip(indices,idxtime):
        if p[1] < besttime[-1]:
            bestidx.append(p[0])
            besttime.append(p[1])
    bestidx.append(idxlen-1)
    besttime.append(besttime[-1])



    fig,ax=plt.subplots(figsize=(10,3.7),tight_layout=True)
    ax.set_ylim(ymin=0,ymax=max(idxtime)*1.1)

    
    alla = ax.scatter(indices, idxtime, marker='x', color='gray',label="All experiments")
    bestonly, = ax.plot(bestidx,besttime,marker='x',color='red',markevery=range(len(bestidx)-1),label='New best')
    baseline, = ax.plot([0,indices[-1]],[idxtime[0],idxtime[0]],marker='x',color='blue',markevery=[0],label='Baseline')  

    ax.set_xlabel("Experiment number")
    ax.set_ylabel("Runtime [s]")
    ax.grid(linestyle='--',axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.annotate(f"{idxtime[0]:.2f} s",xy=(idxlen,idxtime[0]),xytext=(10,0),textcoords='offset points',horizontalalignment='left',verticalalignment='center')
    ax.annotate(f"{besttime[-1]:.2f} s",xy=(idxlen,besttime[-1]),xytext=(10,0),textcoords='offset points',horizontalalignment='left',verticalalignment='center')

    ax.legend(loc='best',handles=[baseline,alla,bestonly])

    imgname = filename.parent / (filename.parent.parent.name + '_' + filename.parent.name + '.pdf')
    plt.savefig(fname=imgname)
    #plt.show()


curdir = pathlib.Path('.')
for f in curdir.glob('**/experiments.csv'):
    makeplot(f)