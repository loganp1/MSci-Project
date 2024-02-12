# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 15:59:01 2024

@author: logan
"""

'''
This file will be to perform the analysis on the split data and get cross correlation data vs various parameters
'''

# Import modules
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
from cross_correlation import cross_correlation

#%% Upload data

df_ACE = pd.read_csv('ace_data_unix.csv')
df_DSCOVR = pd.read_csv('dscovr_data_unix.csv')
df_Wind = pd.read_csv('wind_data_unix.csv')
df_SYM = pd.read_csv('SYM_data_unix.csv')
split_times = np.load('split_data_times.npy')

#%% Split dataframes up into 4 hour periods to analyse

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
    
#%% Record max cross-correlation & zero value CC for every 'sub' df forecast

# Multi vs Real
# MvsR_zvCC = []
# MvsR_maxCC = []
# MvsR_deltaTs = []

# ACE vs Real
# AvsR_zvCC = []
# AvsR_maxCC = []
# AvsR_deltaTs = []

# DSCOVR vs Real
# DvsR_zvCC = []
# DvsR_maxCC = []
# DvsR_deltaTs = []

# ACE vs DSCOVR
AvsD_zvCC = []
AvsD_maxCC = []
AvsD_deltaTs = []


for i in range(len(ace_dfs)):
    
    sc_dict = {'ACE': ace_dfs[i], 'DSCOVR': dsc_dfs[i], 'Wind': wnd_dfs[i]}
    # We plug in the full sym as don't want to remove important outside-edge data 
    myclass = Space_Weather_Forecast(SC_dict=sc_dict, SYM_real=df_SYM)
    tm, t1, t2, t3, sym_forecast1, sym_forecast2, sym_forecast3, sym_forecast_mul = myclass.Compare_Forecasts('both')

    sym_real = myclass.GetSYMdata()['SYM/H, nT']
    treal = myclass.GetSYMdata()['Time']
    
    # Turn series' into lists so we can index properly
    tm, t1, t2, t3, treal, sym_real = (tm.tolist(), t1.tolist(), t2.tolist(), t3.tolist(), treal.tolist(), 
                                       sym_real.tolist())
    
    # Multi vs Real
    
    # Form 2d lists for time series' + data
    mul_list = [tm,sym_forecast_mul]
    real_list = [treal,sym_real]

    common_time, sym_forecast_mulA, sym_realA = align_and_interpolate_datasets(mul_list,real_list,len(tm))
    sym_forecast_mulA, sym_realA = sym_forecast_mulA[0], sym_realA[0]
    
    # Cross-correlation
    time_delays,cross_corr_values = cross_correlation(sym_forecast_mulA, sym_realA, common_time)

    # Find the index where time_delays is 0
    zero_delay_index = np.where(time_delays == 0)[0]
    max_index = np.argmax(cross_corr_values)
    
    deltaT = time_delays[max_index] - time_delays[zero_delay_index]
    
    # Output max CC and zero value CC
    maxCC = max(cross_corr_values)
    zeroValCC = cross_corr_values[zero_delay_index][0]
    
    AvsD_zvCC.append(zeroValCC)
    AvsD_maxCC.append(maxCC)
    AvsD_deltaTs.append(deltaT)
    print(i)
    
#%% Transform deltaTs into list if needed

AvsD_deltaTs = [arr[0] for arr in AvsD_deltaTs]
    
#%% Plot results

plt.hist(AvsD_zvCC)
plt.xlabel('Zero Value Cross Correlation')
plt.ylabel('Frequency')
plt.show()

plt.hist(AvsD_maxCC)
plt.xlabel('Maximum Cross Correlation')
plt.ylabel('Frequency')
plt.show()
    
plt.hist(AvsD_deltaTs)
plt.xlabel('$\Delta$T')
plt.ylabel('Frequency')
plt.show()

#%% Save data

np.save('AvsD_zvCC.npy',AvsD_zvCC)
np.save('AvsD_maxCC.npy',AvsD_maxCC)
np.save('AvsD_deltaTs.npy',AvsD_deltaTs)



