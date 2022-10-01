# -*- coding: utf-8 -*-
"""
Created on Thu May 12 22:34:43 2022

@author: potat
"""

import numpy as np

s = lambda n : ((4000000/n)**(1/3))
t = lambda n : (400 - s(n))/2
f = lambda s,t,n : 10*n + 0.012*s**2 + (((9*10**-3)*n*s**2) / (t+1.5)) + (4*10**-4)*(3*s**2 + 3*s*t + t**2)

arr = np.ones((10000-1,2))
opt = 0

for idx,j in enumerate(range(1,10000)):
    s_val = s(j)
    t_val = t(j)
    f_val = f(s_val, t_val, j)
    
    arr[idx] = [j,f_val]

optimal_n = arr[np.where(arr == np.min(arr[:,1]))[0]][:,0]
print(s(optimal_n))
print(t(optimal_n))

