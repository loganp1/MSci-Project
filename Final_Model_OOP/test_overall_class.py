# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 12:45:50 2024

@author: logan
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add path to the directory containing classes
sys.path.append(r"C:\Users\logan\OneDrive - Imperial College London\Uni\Year 4\MSci Project\MSci-Project"
                r"\Final_Model_OOP")

from Space_Weather_Forecasting_CLASS import Space_Weather_Forecast


#%%

# Load data
df_ACE = pd.read_csv('ace_T4_1min_unix.csv')
df_DSCOVR = pd.read_csv('dscovr_T2_1min_unix.csv')
df_Wind = pd.read_csv('wind_T2_1min_unix.csv')
df_SYM = pd.read_csv('SYM2_unix.csv')

# Create dictionary of dataframes
sc_dict = {'ACE': df_ACE, 'DSCOVR': df_DSCOVR, 'Wind': df_Wind}

# Load this into Space_Weather_Forecast and watch the magic work!
myclass = Space_Weather_Forecast(SC_dict=sc_dict, SYM_real=df_SYM)

#%% Test date filter function & DateTime converter

myclass.unix_to_DateTime()

start_date = '2017-01-01 00:00:00'
end_date = '2017-02-01 00:00:00'

#myclass.filter_dates(start_date,end_date)

#%% Testing - All 4 prediction methods

time_series, sym_forecast1, sym_forecast2, sym_forecast3, sym_forecast_mul = myclass.Compare_Forecasts('both')

#%% 

plt.plot(time_series, sym_forecast1,label='ACE')
plt.plot(time_series, sym_forecast2,label='DSCOVR')
plt.plot(time_series, sym_forecast3,label='Wind')
plt.plot(time_series, sym_forecast_mul,label='Multi')

plt.legend()


#%% Testing - All 3 single spacecraft methods

time_series, sym_forecast1, sym_forecast2, sym_forecast3 = myclass.Compare_Forecasts('single')

#%%

plt.plot(time_series, sym_forecast1,label='ACE')
plt.plot(time_series, sym_forecast2,label='DSCOVR')
plt.plot(time_series, sym_forecast3,label='Wind')

plt.legend()


#%% Testing - Single spacecraft

time_series, sym_forecast = myclass.Compare_Forecasts('single','ACE')

#%% 

plt.plot(time_series, sym_forecast,label='ACE')

plt.legend()


#%% Check where nans exist in NON-filtered ACE data as seems to be loads in ACE

nan_info = myclass.check_nan_values('ACE')