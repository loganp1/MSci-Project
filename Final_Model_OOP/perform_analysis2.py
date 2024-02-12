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
from align_and_interpolate import align_and_interpolate_datasets

#%% Upload data

df_ACE = pd.read_csv('ace_data_unix.csv')
df_DSCOVR = pd.read_csv('dscovr_data_unix.csv')
df_Wind = pd.read_csv('wind_data_unix.csv')
df_SYM = pd.read_csv('SYM_data_unix.csv')
split_times = np.load('split_data_times.npy')

#%%

sc_dict = {'ACE': df_ACE, 'DSCOVR': df_DSCOVR, 'Wind': df_Wind}

myclass = Space_Weather_Forecast(SC_dict=sc_dict, SYM_real=df_SYM)

#%%

myclass.SplitTimes(split_times)

#%%

zvCCs, maxCCs, deltaTs = myclass.GetCC(['multi','real'])

#%%

plt.hist(zvCCs)
plt.xlabel('Zero Time Shift Cross Correlations')
plt.ylabel('Frequency')

