# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 15:07:24 2024

@author: logan
"""

import pandas as pd
import numpy as np
from filter_good_data import modify_list
import matplotlib.pyplot as plt

#%% Set period T to split data into

T = 4

#%%

# Load data
df_ACE = pd.read_csv('ace_data_unix.csv')
# df_DSCOVR = pd.read_csv('dscovr_data_unix.csv')
# df_Wind = pd.read_csv('wind_data_unix.csv')
# df_SYM = pd.read_csv('SYM_data_unix.csv')

#%%

df_ACE['DateTime'] = pd.to_datetime(df_ACE['Time'], unit='s')
# df_DSCOVR['DateTime'] = pd.to_datetime(df_DSCOVR['Time'], unit='s')
# df_Wind['DateTime'] = pd.to_datetime(df_Wind['Time'], unit='s')
# df_SYM['DateTime'] = pd.to_datetime(df_SYM['Time'], unit='s')

#%% Now split the data based on the max 10 nans in a row filter for ACE (as density data so poor)
# NOPE, now filtered for ALL data (all 9, 3 each SC)
T=4
min_max_times = np.load('max10_NaNs_ALLdata_start_end_times.npy')

# Apply Ned's function to split data into 4 hour periods

split_times = modify_list(min_max_times,T*3600,T*3600)

np.save('split_data_times_ALLf.npy', split_times)

#%%

# Initialize an empty list to store the split DataFrames
split_dfs = []

# Split the DataFrame based on the min_max_times
for i in range(len(split_times)):
    start_time, end_time = split_times[i]
    subset_df_ACE = df_ACE[(df_ACE['Time'] >= start_time) & (df_ACE['Time'] <= end_time)].copy()
    split_dfs.append(subset_df_ACE)
    
    
#%% Excellent, now have a list of suitable dataframes, now test timeframe of each

subdf_lens = []

for df in split_dfs:
    
    subdf_lens.append(len(df)/60)
    
plt.hist(subdf_lens,bins=20)
plt.xlabel('Time Length of Subset (Hours)')    
plt.ylabel('Frequency')