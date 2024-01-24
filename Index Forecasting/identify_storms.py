# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:00:58 2024

@author: logan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read in SYM/H index & plasma data
df_params = pd.read_csv('omni_ind_params_2019_1min.csv')


#%% Clean Data

# Identify rows with NaN values
df_params['BZ, nT (GSE)'] = df_params['BZ, nT (GSE)'].replace(9999.99, np.nan)
df_params['Speed, km/s'] = df_params['Speed, km/s'].replace(99999.9, np.nan)
df_params['Flow pressure, nPa'] = df_params['Flow pressure, nPa'].replace(99.99, np.nan)
df_params['Electric field, mV/m'] = df_params['Electric field, mV/m'].replace(999.99, np.nan)
df_params['Proton Density, n/cc'] = df_params['Proton Density, n/cc'].replace(999.9, np.nan)

# Interpolate NaN valuesind_params_2001_1min.csv'
for column in df_params.columns[3:]:  # Exclude the time columns
    df_params[column] = df_params[column].interpolate()

# Drop rows with remaining NaN values
df_params = df_params.dropna()


#%% Create time series and isolate different columns

# Combine 'Day' and 'Hour' into a new column 'DayHour'
df_params['DayHourMin'] = df_params['Day'] + df_params['Hour'] / 24.0 + df_params['Minute'] / 60 / 24
time = df_params['DayHourMin']

# Name the different datasets
sym = df_params['SYM/H, nT']
Bz = df_params['BZ, nT (GSE)']
speed = df_params['Speed, km/s']
pressure = df_params['Flow pressure, nPa']
Efield = df_params['Electric field, mV/m']
density = df_params['Proton Density, n/cc']

#%% Identify storm events

plt.plot(time,sym)


