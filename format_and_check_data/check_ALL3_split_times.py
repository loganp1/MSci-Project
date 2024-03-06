# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 01:01:04 2024

@author: logan
"""

import numpy as np
import pandas as pd

times = np.load('split_data_times_ALLf.npy')
SYM = pd.read_csv('SYM_data_unix.csv')

symDisp=[]
for i in range(len(times)):
    print(i)
    dfFilt = SYM[(SYM['Time'] >= times[i][0]) & (SYM['Time'] <= times[i][1])]
    symDisp.append(abs(sum( x for x in dfFilt['SYM/H, nT'] if x<0)))
    
    
#%%

plt.hist(symDisp)
