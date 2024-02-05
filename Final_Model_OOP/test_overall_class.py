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
df_ACE = pd.read_csv('ace_T2_1min_unix.csv')
df_DSCOVR = pd.read_csv('dscovr_T2_1min_unix.csv')
df_Wind = pd.read_csv('wind_T2_1min_unix.csv')
df_SYM = pd.read_csv('SYM2_unix.csv')

# Create dictionary of dataframes
sc_dict = {'ACE': df_ACE, 'DSCOVR': df_DSCOVR, 'Wind': df_Wind}

# Load this into Space_Weather_Forecast and watch the magic work!
myclass = Space_Weather_Forecast(SC_dict=sc_dict, SYM_real=df_SYM)

#%% Now propagate choosing which method/spacecraft! 

sym_forecast, time_series = myclass.Forecast_SYM_H('multi')

#%% Plot results

plt.plot(time_series, sym_forecast)

