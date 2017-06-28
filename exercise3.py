# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:54:55 2017

@author: dan
"""
#%%
days = 0
bills = 1
 
while bills < 32767:
    bills = bills*2
    days += 1

print("number of days =", days)
print("number of bills =", bills)



