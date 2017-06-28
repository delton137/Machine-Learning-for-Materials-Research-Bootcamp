#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:07:58 2017

@author: dan 
"""
#%%
names = ['alex', 'dan', 'giliad']
colors = ['green', 'blue', 'red']

print([i for i in enumerate(names)])

for (names, colors) in zip(names, colors):

    
#%% Copy vs Deep copy

a = [1,2,3]
b = a
c = a.append(4)
print(b)

import copy 
f = copy.deepcopy(a)

a.append(1000)
print(f)

import numpy as np