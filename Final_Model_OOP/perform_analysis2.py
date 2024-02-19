# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 12:24:23 2024

@author: logan
"""

import sys
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Add path to the directory containing classes
sys.path.append(r"C:\Users\logan\OneDrive - Imperial College London\Uni\Year 4\MSci Project\MSci-Project"
                r"\Final_Model_OOP")

from Space_Weather_Forecasting_CLASS import Space_Weather_Forecast

#%% Upload data

df_ACE = pd.read_csv('ace_data_unix.csv')
df_DSCOVR = pd.read_csv('dscovr_data_unix.csv')
df_Wind = pd.read_csv('wind_data_unix.csv')
df_SYM = pd.read_csv('SYM_data_unix.csv')
split_times = np.load('split_data_times.npy')
df_OMNI = pd.read_csv('OMNI_EP_data_unix.csv')

#%% OMNI data needs to be cleaned

df_OMNI['DateTime'] = pd.to_datetime(df_OMNI['Time'], unit='s')

df_OMNI['Pressure'] = df_OMNI['Flow pressure, nPa'].replace(99.99, np.nan)
df_OMNI['Efield'] = df_OMNI['Electric field, mV/m'].replace(999.99, np.nan)

for column in df_OMNI.columns:
    df_OMNI[column] = df_OMNI[column].interpolate()

# Drop rows with remaining NaN values
df_OMNI = df_OMNI.dropna()

#%% Create dictionary of data and initialise main class for forecasting

sc_dict = {'ACE': df_ACE, 'DSCOVR': df_DSCOVR, 'Wind': df_Wind}

# TRY NOT TO CALL CLASS SAME AS SOMETHING IN CLASSES (NOT SURE EXACTLY IF IT MATTERS BUT MAY CAUSE ISSUES)

MYclass = Space_Weather_Forecast(SC_dict=sc_dict, SYM_real=df_SYM, OMNI_data=df_OMNI)

#%% Split data into 4 hour periods

# DON'T RUN THIS AGAIN ONLY ONCE, RELOAD DATA IF DO ACCIDENTALLY
MYclass.SplitTimes(split_times)

#%% Get data

zvCCs, maxCCs, deltaTs = MYclass.GetCC(['multi','real'])

#%% Save data files

np.save('OvsR_zvCCs_d15.npy', zvCCs)
np.save('OvsR_maxCCs_d15.npy', maxCCs)
np.save('OvsR_deltaTs_d15.npy', deltaTs)

#%% Plot data

plt.hist(zvCCs)
plt.xlabel('Zero Time Shift Cross Correlations')
plt.ylabel('Frequency')
plt.show()

plt.hist(deltaTs)

#%% Test sub dfs are correct after splitting

data = MYclass.GetSCsubDFs()

#%% I want to test one of the zero (or a few) deltaT values, so let's just forecast those periods
# There are zeros at indices 5,6,7

test_MYclass = Space_Weather_Forecast(SC_dict=sc_dict, SYM_real=df_SYM)

test_zvCCs, test_maxCCs, test_deltaTs = MYclass.GetCC(['ACE','DSCOVR'])


#%% Create comparison histogram



