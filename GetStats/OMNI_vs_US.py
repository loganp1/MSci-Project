# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:36:19 2024

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

from SYM_H_Model_CLASS import SYM_H_Model
from SC_Propagation_CLASS import SC_Propagation
from Space_Weather_Forecasting_CLASS import Space_Weather_Forecast

#%% Firstly, my propagation results

# Import data
df_ACE = pd.read_csv('ace_data_unix.csv')
df_DSCOVR = pd.read_csv('dscovr_data_unix.csv')
df_Wind = pd.read_csv('wind_data_unix.csv')
df_SYM = pd.read_csv('SYM_data_unix.csv')

#%%

# Create dictionary of dataframes
sc_dict = {'ACE': df_ACE, 'DSCOVR': df_DSCOVR, 'Wind': df_Wind}

# Load this into Space_Weather_Forecast and watch the magic work!
classa = Space_Weather_Forecast(SC_dict=sc_dict, SYM_real=df_SYM)

# Lets do a forecast between these dates 

classa.unix_to_DateTime()

#start_date = '2018-08-22 00:00:00'
#end_date = '2018-08-30 00:00:00'

#start_date = '2018-03-28 10:59:00'
#end_date = '2018-05-03 14:18:00'

start_date = '2018-03-28 10:59:00'
end_date = '2018-04-03 14:18:00'


# Convert strings to datetime objects
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

classa.filter_dates(start_date,end_date)

df_ACEfilt = classa.GetSCdata('ACE')
df_DSCfilt = classa.GetSCdata('DSCOVR')
df_Wndfilt = classa.GetSCdata('Wind')

#%%

prop_class = SC_Propagation(classa.GetSCdf())
prop_class.required_form()

# Estimating sym0 value by eye
sym0 = -8

sym_multi, time_multi = classa.Forecast_SYM_H(sym0, prop_class, 'multi')

#%%

plt.plot(time_multi,sym_multi)

#%% Now forecast using E & P from OMNI

EPdf = pd.read_csv('OMNI_EP_data_unix.csv')
EPdf['DateTime'] = pd.to_datetime(EPdf['Time'], unit='s')

EPdf['Pressure'] = EPdf['Flow pressure, nPa'].replace(99.99, np.nan)
EPdf['Efield'] = EPdf['Electric field, mV/m'].replace(999.99, np.nan)

for column in EPdf.columns:
    EPdf[column] = EPdf[column].interpolate()

# Drop rows with remaining NaN values
EPdf = EPdf.dropna()

# Filter Dates
mask = (EPdf['DateTime'] >= start_date) & (EPdf['DateTime'] <= end_date)
EPdf = EPdf[mask].copy()

#%%

EPdf = EPdf.reset_index()
sym_class = SYM_H_Model(EPdf,sym_multi[0])

sym_OMNI = sym_class.predict_SYM()
time_OMNI = EPdf['Time']

#%%

import matplotlib.dates as mdates

time_real = classa.GetSYMdata()['DateTime']
sym_real = classa.GetSYMdata()['SYM/H, nT']

time_multi = pd.to_datetime(time_multi,unit='s')
time_OMNI = pd.to_datetime(time_OMNI,unit='s')

plt.plot(time_real,sym_real,label='Real SYM/H',linewidth=1,color='grey')
plt.plot(time_multi,sym_multi,label='Multi Spacecraft Prediction',linewidth=1)
plt.plot(time_OMNI,sym_OMNI,label='OMNI E & P Predictions',linewidth=1)
plt.legend(fontsize=9)
plt.grid()
plt.xlabel('Date')
plt.ylabel('SYM/H (nT)')

plt.xticks(rotation=0, 
           ticks=pd.date_range(start=time_real.min(), end=time_real.max(), freq='1D'))  # Set explicit ticks

# Use DateFormatter to customize the format of dates
date_format = mdates.DateFormatter('%d-%m')
plt.gca().xaxis.set_major_formatter(date_format)
plt.grid()


