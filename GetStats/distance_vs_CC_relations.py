# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:10:05 2024

@author: logan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Upload data

df_ACE = pd.read_csv('ace_data_unix.csv')
df_DSCOVR = pd.read_csv('dscovr_data_unix.csv')
df_Wind = pd.read_csv('wind_data_unix.csv')
df_SYM = pd.read_csv('SYM_data_unix.csv')
split_times = np.load('split_data_times.npy')

#%%

# Initialize empty lists to store the split DataFrames
ace_dfs = []
dsc_dfs = []
wnd_dfs = []

# Split the DataFrame based on the min_max_times
for i in range(len(split_times)):
    start_time, end_time = split_times[i]
    subset_df_ACE = df_ACE[(df_ACE['Time'] >= start_time) & (df_ACE['Time'] <= end_time)].copy()
    ace_dfs.append(subset_df_ACE)
    subset_df_DSC = df_DSCOVR[(df_DSCOVR['Time'] >= start_time) & (df_DSCOVR['Time'] <= end_time)].copy()
    dsc_dfs.append(subset_df_DSC)
    subset_df_Wind = df_Wind[(df_Wind['Time'] >= start_time) & (df_Wind['Time'] <= end_time)].copy()
    wnd_dfs.append(subset_df_Wind)
    
    
    
#%% ACE yz offset from origin

Ayz_offset = []

for i in range(len(ace_dfs)):
    
    sepvec = np.array([[np.mean(ace_dfs[i]['y'])],
                      [np.mean(ace_dfs[i]['z'])]])
    
    sepval = np.linalg.norm(sepvec)
    Ayz_offset.append(sepval)    
    
#%% ACE - DSCOVR separation distance

ADsep = []

for i in range(len(ace_dfs)):
    
    sepvec = np.array([[np.mean(ace_dfs[i]['x'] - dsc_dfs[i]['Wind, Xgse,Re'])],
                      [np.mean(ace_dfs[i]['y'] - dsc_dfs[i]['Wind, Ygse,Re'])],
                      [np.mean(ace_dfs[i]['z'] - dsc_dfs[i]['Wind, Zgse,Re'])]])
    
    sepval = np.linalg.norm(sepvec)
    ADsep.append(sepval)
    
#%% ACE - DSCOVR yz offset
    
ADyz_offset = []
                       
for i in range(len(ace_dfs)):
    
    sepvec = np.array([[np.mean(ace_dfs[i]['y'] - dsc_dfs[i]['Wind, Ygse,Re'])],
                      [np.mean(ace_dfs[i]['z'] - dsc_dfs[i]['Wind, Zgse,Re'])]])
    
    sepval = np.linalg.norm(sepvec)
    ADyz_offset.append(sepval)
    

#%%

AvsD_zvCC = np.load('AvsD_zvCCs.npy')
# DvsR_zvCC = np.load('DvsR_zvCCs.npy')
# MvsR_zvCC = np.load('MvsR_zvCCs.npy')
# AvsW_zvCC = np.load('AvsW_zvCCs.npy')

plt.scatter(ADyz_offset,AvsD_zvCC,marker='x')
plt.xlabel('Y-Z Offset')
plt.ylabel('Cross Correlation with Measured SYM/H')

# plt.scatter(AWyz_offset,AvsD_zvCC,marker='x')
# plt.xlabel('Y-Z Offset')
# plt.ylabel('Cross Correlation with Measured SYM/H')

#%%

plt.hist(MvsR_zvCC)
plt.show()

plt.hist(AvsR_zvCC)
plt.show()

plt.hist(DvsR_zvCC)
plt.show()

comp_diffAD = np.asarray(AvsR_zvCC) - np.asarray(DvsR_zvCC)
