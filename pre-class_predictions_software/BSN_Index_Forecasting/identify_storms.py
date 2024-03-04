# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:00:58 2024

@author: logan
"""

'''This file will locate storms and therefore useful analysis periods using SYM/H'''

import pandas as pd
import matplotlib.pyplot as plt

# Read in SYM/H index & plasma data
df_params1 = pd.read_csv('SYM1.csv')
df_params2 = pd.read_csv('SYM2.csv')
df_params3 = pd.read_csv('SYM3.csv')
df_params4 = pd.read_csv('SYM4.csv')

#%% Create time series and sym variable

# Combine 'Year', 'Day', 'Hour', and 'Minute' into a new column 'Datetime'
df_params1['Datetime'] = pd.to_datetime(df_params1['Year'].astype(str) + df_params1['Day'].astype(str).str.zfill(3) 
                                        + df_params1['Hour'].astype(str).str.zfill(2) 
                                        + df_params1['Minute'].astype(str).str.zfill(2), format='%Y%j%H%M')

time1 = df_params1['Datetime']

# Name the different datasets
sym1 = df_params1['SYM/H, nT']

# Identify storm events
plt.plot(time1, sym1)




#%% Create time series and sym variable for the New Dataset 2

# Combine 'Day', 'Hour', and 'Minute' into a new column 'Datetime'
df_params2['Datetime'] = pd.to_datetime(df_params2['Year'].astype(str) + df_params2['Day'].astype(str).str.zfill(3) 
                                        + df_params2['Hour'].astype(str).str.zfill(2) 
                                        + df_params2['Minute'].astype(str).str.zfill(2), format='%Y%j%H%M')
time2 = df_params2['Datetime']

# Name the different datasets for the New Dataset 2
sym2 = df_params2['SYM/H, nT']

# Identify storm events for the New Dataset 2
plt.plot(time2, sym2)


#%% Create time series and sym variable for the New Dataset 3

# Combine 'Day', 'Hour', and 'Minute' into a new column 'Datetime'
df_params3['Datetime'] = pd.to_datetime(df_params3['Year'].astype(str) + df_params3['Day'].astype(str).str.zfill(3) 
                                        + df_params3['Hour'].astype(str).str.zfill(2) 
                                        + df_params3['Minute'].astype(str).str.zfill(2), format='%Y%j%H%M')
time3 = df_params3['Datetime']

# Name the different datasets for the New Dataset 3
sym3 = df_params3['SYM/H, nT']

# Identify storm events for the New Dataset 3
plt.plot(time3, sym3)


 #%% Create time series and sym variable for the New Dataset 4

# Combine 'Day', 'Hour', and 'Minute' into a new column 'Datetime'
df_params4['Datetime'] = pd.to_datetime(df_params4['Year'].astype(str) + df_params4['Day'].astype(str).str.zfill(3) 
                                        + df_params4['Hour'].astype(str).str.zfill(2) 
                                        + df_params4['Minute'].astype(str).str.zfill(2), format='%Y%j%H%M')
time4 = df_params4['Datetime']

# Name the different datasets for the New Dataset 4
sym4 = df_params4['SYM/H, nT']

# Identify storm events for the New Dataset 4
plt.plot(time4, sym4)


#%%

df = pd.read_csv('SYM_data_unix.csv')
time = pd.to_datetime(df['Time'], unit='s')
sym = df['SYM/H, nT']
time[0] = pd.to_datetime('2016-06-28 00:00:00.111111111')  # Checking decimals work for other file as having trouble
#%%
plt.plot(time, sym)

# Customize x-axis ticks and labels
plt.xticks(rotation=45, ha='right')  # Rotate the labels for better visibility

plt.show()

#%%

max_index = sym.idxmin()
