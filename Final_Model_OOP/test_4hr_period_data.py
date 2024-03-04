# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:11:22 2024

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


#%%

# Load data
df_ACE = pd.read_csv('ace_data_unix.csv')
df_DSCOVR = pd.read_csv('dscovr_data_unix.csv')
df_Wind = pd.read_csv('wind_data_unix.csv')
df_SYM = pd.read_csv('SYM_data_unix.csv')

#%%

df_ACE['DateTime'] = pd.to_datetime(df_ACE['Time'], unit='s')

#%% Now split the data based on the max 10 nans in a row filter for ACE (as density data so poor)

min_max_times = np.load('ace_nNaNs_max10_times.npy')

# Convert min_max_times to datetime objects
min_max_times = pd.to_datetime(min_max_times,unit='s')

# Filter so we don't have time periods less than 4 hours
valid_ranges = [i for i in range(len(min_max_times)) if (min_max_times[i][1] - min_max_times[i][0]).total_seconds()
                >= 4 * 3600]
min_max_times = min_max_times.to_numpy()[valid_ranges]

# Initialize an empty list to store the split DataFrames
split_dfs = []

# Split the DataFrame based on the min_max_times
for i in range(len(min_max_times)):
    start_time, end_time = min_max_times[i]
    subset_df_ACE = df_ACE[(df_ACE['DateTime'] >= start_time) & (df_ACE['DateTime'] <= end_time)].copy()
    split_dfs.append(subset_df_ACE)
    
    
#%% Excellent, now have a list of suitable dataframes, now test timeframe of each

subdf_lens = []

for df in split_dfs:
    
    subdf_lens.append(len(df)/60)
    
plt.hist(subdf_lens,bins=20)
plt.xlabel('Time Length of Subset (Hours)')    
plt.ylabel('Frequency')

#%% Rerun this every time after running below test otherwise will X pos data by Re each time from required_form()

# Create dictionary of dataframes
sc_dict = {'ACE': df_ACE, 'DSCOVR': df_DSCOVR, 'Wind': df_Wind}

# Load this into Space_Weather_Forecast and watch the magic work!
myclass = Space_Weather_Forecast(SC_dict=sc_dict, SYM_real=df_SYM)

#%% Testing - All 4 prediction methods

tm, t1, t2, t3, sym_forecast1, sym_forecast2, sym_forecast3, sym_forecast_mul = myclass.Compare_Forecasts('both')

#%% 

sym_real = myclass.GetSYMdata()['SYM/H, nT']
time_series = myclass.GetSYMdata()['Time']

time_series = pd.to_datetime(time_series,unit='s')

# Define a custom set of distinct colors
custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
l = 1

import matplotlib.dates as mdates

# Assuming sym_forecast1, sym_forecast2, sym_forecast3, sym_forecast_mul are your forecasted data
plt.plot(t1, sym_forecast1, label='ACE', color=custom_colors[0], linewidth=l)
plt.plot(t2, sym_forecast2, label='DSCOVR', color=custom_colors[1], linewidth=l)
plt.plot(t3, sym_forecast3, label='Wind', color=custom_colors[2], linewidth=l)
plt.plot(tm, sym_forecast_mul, label='Multi', color=custom_colors[3], linewidth=l)
plt.plot(time_series, sym_real, label='SYM/H', color=custom_colors[4], linewidth=l)

plt.xticks(rotation=0, 
           ticks=pd.date_range(start=time_series.min(), end=time_series.max(), freq='5D'))  # Set explicit ticks

# Use DateFormatter to customize the format of dates
date_format = mdates.DateFormatter('%d-%m')
plt.gca().xaxis.set_major_formatter(date_format)

plt.grid()
plt.legend()

plt.xlabel('Date in 2018')
plt.ylabel('SYM/H (nT)')

plt.tight_layout()
plt.savefig('storm_data_range_forecast.png',dpi=500)
plt.show()