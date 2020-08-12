#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 22:19:49 2019

@author: victor
"""

#%%

import pandas as pd
import numpy as np


#%% Lecture 3: small grid, iterative policy evaluation

grid = pd.DataFrame(np.zeros([4,4]))


k = 0

def update(grid, k):
    terminal = [[0,0], [3,3]]
    tmp = pd.DataFrame(np.zeros([4,4]))
    for i in range(4):
        for j in range(4):
            if [i,j] not in terminal:
                tmp.iloc[i,j] = 0.25*(-1 + grid.iloc[np.max([0,i-1]), j]) + \
                                0.25*(-1 + grid.iloc[np.min([3,i+1]), j]) + \
                                0.25*(-1 + grid.iloc[i, np.max([0,j-1])]) + \
                                0.25*(-1 + grid.iloc[i, np.min([3,j+1])]) 
    k += 1
    print('\n')
    print('k =:',k)
    print(tmp)
    print('_'*50)
    return tmp, k


while k < 3:
    grid, k = update(grid, k)
    
    
#%% Lecture 3: shortest path, value iteration


grid = pd.DataFrame(np.zeros([4,4]))


k = 1


def update(grid, k):
    tmp = pd.DataFrame(np.zeros([4,4]))
    terminal = [[0,0]]
    for i in range(4):
        for j in range(4):
            if [i,j] not in terminal:
                n = grid.iloc[np.max([0,i-1]), j]
                s = grid.iloc[np.min([3,i+1]), j]
                w = grid.iloc[i, np.max([0,j-1])]
                e = grid.iloc[i, np.min([3,j+1])] 
                tmp.iloc[i,j] = -1 + np.max([n,s,w,e])
                          
    k += 1
    print('\n')
    print('k =:',k)
    print(tmp)
    print('_'*50)
    return tmp, k


i = 0
while i < 7:
    grid, k = update(grid, k)
    i += 1
    
    
    
