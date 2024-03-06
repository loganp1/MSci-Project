# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:08:05 2024

@author: logan
"""

import numpy as np
import pandas as pd
times=np.load("split_data_times.npy")

time1,time2 = times[2584]

import matplotlib.pyplot as plt

df_ACE = pd.read_csv('ace_data_unix.csv')

# Filter the data for the specified time range
filtered_data = df_ACE[(df_ACE['Time'] >= time1) & (df_ACE['Time'] <= time2)]

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(filtered_data['Time'], filtered_data['Bz'], label='Bz')
plt.title('Bz vs. Time')
plt.xlabel('Time')
plt.ylabel('Bz')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(filtered_data['Time'], filtered_data['vx'], label='vx', color='orange')
plt.title('vx vs. Time')
plt.xlabel('Time')
plt.ylabel('vx')
plt.legend()

plt.tight_layout()
plt.show()

#%%

dfs = []

for i in range(len(times)):
    print(i)
    dfFilt = df_ACE[(df_ACE['Time'] >= times[i][0]) & (df_ACE['Time'] <= times[i][1])]
    dfs.append(dfFilt)
    
#%%

nan_counts = []

for df in dfs:
    
    nan_counts.append(df['Bz'].isna().sum())
    
    
    
#%% ############################################################

# Repeat all of above for new split times which includes Bz and v nans as well as density, n

times=np.load("split_data_times_new.npy")

time1,time2 = times[2584]

import matplotlib.pyplot as plt

df_ACE = pd.read_csv('ace_data_unix.csv')

# Filter the data for the specified time range
filtered_data = df_ACE[(df_ACE['Time'] >= time1) & (df_ACE['Time'] <= time2)]

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(filtered_data['Time'], filtered_data['Bz'], label='Bz')
plt.title('Bz vs. Time')
plt.xlabel('Time')
plt.ylabel('Bz')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(filtered_data['Time'], filtered_data['vx'], label='vx', color='orange')
plt.title('vx vs. Time')
plt.xlabel('Time')
plt.ylabel('vx')
plt.legend()

plt.tight_layout()
plt.show()

#%%

dfs = []

for i in range(len(times)):
    print(i)
    dfFilt = df_ACE[(df_ACE['Time'] >= times[i][0]) & (df_ACE['Time'] <= times[i][1])]
    dfs.append(dfFilt)
    
#%%

nan_counts = []

for df in dfs:
    
    nan_counts.append(df['Bz'].isna().sum())
