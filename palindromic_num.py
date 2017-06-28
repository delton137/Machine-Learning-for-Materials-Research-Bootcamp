#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:52:34 2017

@author: dan
"""
#%% Method 1
def check_palindromic(num):
    string = str(num)
    if string == string[::-1]:
        return True

greatest = 0  
for i in range(100,1001):
    for j in range(i,1001):
        num = i*j
        if (check_palindromic(num) == True): 
            if (num > greatest): 
                greatest = num
            
print(greatest)

#%% Method 2
import numpy as np

a = np.array(range(1000))
b = a 
c = np.meshgrid(a, b)
d = c[0]*c[1]
d = np.resize(d,1000000)
d_str = np.array(d, dtype = str)

y = []
for x in d_str:
    if (x == x[::-1]):
        y = np.append(y,x)
        
final = np.array(y, dtype=int)

print(max(final))
