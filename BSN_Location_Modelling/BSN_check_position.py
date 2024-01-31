# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:43:32 2024

@author: logan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('omni_ind_params_BSN_2023_1min_unix.csv')

df['BSN location, Xgse,Re'] = df['BSN location, Xgse,Re'].replace(9999.99, np.nan)

# Interpolate NaN values
df['BSN location, Xgse,Re'] = df['BSN location, Xgse,Re'].interpolate()
    
    
time = df['Time']
X = df['BSN location, Xgse,Re'].values
    
#%%

plt.plot(time,X)
plt.xlim([time[120000],time[120040]])
plt.show()

plt.hist(X,bins=100)


#%%



for i in range(len(X)-60):
    
    