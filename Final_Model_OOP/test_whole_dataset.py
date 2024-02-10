# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:07:44 2024

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


#%%

# Load data
df_ACE = pd.read_csv('ace_data_unix.csv')
df_DSCOVR = pd.read_csv('dscovr_data_unix.csv')
df_Wind = pd.read_csv('wind_data_unix.csv')
df_SYM = pd.read_csv('SYM_data_unix.csv')

#%% Rerun this every time after running below test otherwise will X pos data by Re each time from required_form()

# Create dictionary of dataframes
sc_dict = {'ACE': df_ACE, 'DSCOVR': df_DSCOVR, 'Wind': df_Wind}

# Load this into Space_Weather_Forecast and watch the magic work!
myclass = Space_Weather_Forecast(SC_dict=sc_dict, SYM_real=df_SYM)

# Lets do a forecast between these dates 

myclass.unix_to_DateTime()

start_date = '2018-08-22 00:00:00'
end_date = '2018-08-30 00:00:00'

#start_date = '2018-03-28 10:59:00'
#end_date = '2018-05-03 14:18:00'

#start_date = '2018-03-28 10:59:00'
#end_date = '2018-04-03 14:18:00'



# Convert strings to datetime objects
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

myclass.filter_dates(start_date,end_date)

df_ACEfilt = myclass.GetSCdata('ACE')
df_DSCfilt = myclass.GetSCdata('DSCOVR')
df_Wndfilt = myclass.GetSCdata('Wind')

#%% Testing - All 4 prediction methods

tm, t1, t2, t3, sym_forecast1, sym_forecast2, sym_forecast3, sym_forecast_mul = myclass.Compare_Forecasts('both')

#%% 

sym_real = myclass.GetSYMdata()['SYM/H, nT']
treal = myclass.GetSYMdata()['Time']

#treal = pd.to_datetime(treal,unit='s')

# Define a custom set of distinct colors
custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
l = 1

import matplotlib.dates as mdates

# Assuming sym_forecast1, sym_forecast2, sym_forecast3, sym_forecast_mul are your forecasted data
plt.plot(t1, sym_forecast1, label='ACE', color=custom_colors[0], linewidth=l)
plt.plot(t2, sym_forecast2, label='DSCOVR', color=custom_colors[1], linewidth=l)
plt.plot(t3, sym_forecast3, label='Wind', color=custom_colors[2], linewidth=l)
plt.plot(tm, sym_forecast_mul, label='Multi', color=custom_colors[3], linewidth=l)
plt.plot(treal, sym_real, label='SYM/H', color=custom_colors[4], linewidth=l)

plt.xticks(rotation=0, 
           ticks=pd.date_range(start=treal.min(), end=treal.max(), freq='5D'))  # Set explicit ticks

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

#%% To do cross correlation we need a common time series

# Turn series' into lists so we can index properly
tm, t1, t2, t3, treal, sym_real = (tm.tolist(), t1.tolist(), t2.tolist(), t3.tolist(), treal.tolist(), 
                                   sym_real.tolist())


#%%

# Form 2d lists for time series' + data
mul_list = [tm,sym_forecast_mul]
real_list = [treal,sym_real]

common_time, sym_forecast_mulA, sym_realA = align_and_interpolate_datasets(mul_list,real_list,len(sym_real))

from cross_correlation import cross_correlation

lags = np.arange(-len(common_time) + 1, len(common_time))
dt = 60

# Calculate corresponding time delays
time_delays = lags * dt

time_delays,cross_corr_values = cross_correlation(sym_forecast_mulA, sym_realA, time_delays)

# Plot the cross-correlation values
plt.figure(figsize=(8, 5))
plt.plot(time_delays/60, cross_corr_values, label='Cross-correlation')
#plt.axvline(time_delays[peak_index]/60, color='red', linestyle='--', label='Peak Correlation')
plt.xlabel('Time Delay (minutes)', fontsize=12)
plt.ylabel('Cross-Correlation', fontsize=12)
plt.title('Cross-Correlation DSCOVR vs Wind Forecasts', fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
#plt.savefig('cross_corr_dscovr_wind.png', dpi=300)
plt.show()

peak_index = np.argmax(np.abs(cross_corr_values))
print('\nPeak cross-correlation between DSCOVR and ACE is observed at a time delay of', 
      int(time_delays[peak_index]/60), 'minutes')

#%%

def Gaussian(x, A, mu, sigma):
    
    return A * np.exp(-(x - mu)**2/(2 * sigma**2))

# Isolate the peak of the cross-correlation curve
lower_lim = -2000*60   # x60 because graph shows in minutes but data is in seconds, so convert mins-->secs
upper_lim = 500*60

# Ensure that the boolean array matches the size of the original arrays
boolean_mask = (time_delays >= lower_lim) & (time_delays <= upper_lim)

# Filter the arrays based on the limits
filtered_cross_corr_values = cross_corr_values[boolean_mask]
filtered_time_delays = time_delays[boolean_mask]

# Plot the cross-correlation values for the filtered data
plt.figure(figsize=(8, 5))
plt.plot(filtered_time_delays/60, filtered_cross_corr_values, label='Filtered Cross-Correlation')

# Find the index of the peak in the filtered data
peak_index = np.argmax(filtered_cross_corr_values)

# Plot the peak in the filtered data
plt.axvline(filtered_time_delays[peak_index]/60, color='red', linestyle='--', label='Peak Correlation')

# Fitting code for the new filtered data
popt, pcov = curve_fit(Gaussian, filtered_time_delays/60, filtered_cross_corr_values)
plt.plot(filtered_time_delays/60, Gaussian(filtered_time_delays/60, *popt), label='Gaussian Fit')

plt.xlabel('Time Delay (minutes)', fontsize=12)
plt.ylabel('Cross-Correlation', fontsize=12)
plt.title('Cross-Correlation DSCOVR vs Wind', fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
#plt.savefig('cross_corr_filtered_dscovr_wind.png', dpi=300)
plt.show()


#%% Compare SYM extrema to forecast extrema

#sym_real = sym_real.tolist()
# Reset indices of time series as otherwise its indices are for the slice in larger df
time_series = time_series.reset_index(drop=True)

#%%

time_min_mul = t1[sym_forecast_mul.index(min(sym_forecast_mul))]
time_min_sym = time_series[sym_real.index(min(sym_real))]

#%% Okay, so peak difference around 30 mins which is ~accurate according to Burton et al.
### So let's try scipy's CC method instead

cc_test = scipy.signal.correlate(sym_forecast_mul, sym_real, mode='full', method='auto')

lags = np.arange(-len(t1) + 1, len(t1))
dt = 60
time_delays = lags * dt

plt.plot(time_delays,cc_test)
