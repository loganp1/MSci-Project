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
import matplotlib.style as style
from matplotlib.ticker import AutoMinorLocator

# Add path to the directory containing classes
sys.path.append(r"C:\Users\logan\OneDrive - Imperial College London\Uni\Year 4\MSci Project\MSci-Project"
                r"\Final_Model_OOP")

from Space_Weather_Forecasting_CLASS import Space_Weather_Forecast
from align_and_interpolate import align_and_interpolate_datasets
from SC_Propagation_CLASS import SC_Propagation


#%%

# Load data
df_ACE = pd.read_csv('ace_data_unix.csv')
df_DSCOVR = pd.read_csv('dscovr_data_unix.csv')
df_Wind = pd.read_csv('wind_data_unix.csv')
df_SYM = pd.read_csv('SYM_data_unix.csv')

#%%

# For testing purposes make much smaller
# df_ACE = pd.read_csv('ace_data_unix.csv').head(1000)
# df_DSCOVR = pd.read_csv('dscovr_data_unix.csv').head(1000)
# df_Wind = pd.read_csv('wind_data_unix.csv').head(1000)
# df_SYM = pd.read_csv('SYM_data_unix.csv').head(1000)


#%% Rerun this every time after running below test otherwise will X pos data by Re each time from required_form()

# Create dictionary of dataframes
sc_dict = {'ACE': df_ACE, 'DSCOVR': df_DSCOVR, 'Wind': df_Wind}

prop_class = SC_Propagation(sc_dict)

# Load this into Space_Weather_Forecast and watch the magic work!
MYclass = Space_Weather_Forecast(SC_dict=sc_dict, SYM_real=df_SYM)

'''
NOTE: CRITICAL!!!!!
We CANNOT name our class myclass as that is used within the classes!
(Not sure this does actually cause the problems after all)
'''

# Lets do a forecast between these dates 

MYclass.unix_to_DateTime()

#start_date = '2018-08-22 00:00:00'
#end_date = '2018-08-30 00:00:00'

#start_date = '2018-03-28 10:59:00'
#end_date = '2018-05-03 14:18:00'

#start_date = '2018-03-28 10:59:00'
#end_date = '2018-04-03 14:18:00'



# Convert strings to datetime objects
#start_date = pd.to_datetime(start_date)
#end_date = pd.to_datetime(end_date)

#MYclass.filter_dates(start_date,end_date)

#df_ACEfilt = myclass.GetSCdata('ACE')
#df_DSCfilt = myclass.GetSCdata('DSCOVR')
#df_Wndfilt = myclass.GetSCdata('Wind')

#%% Testing - All 4 prediction methods

tm, t1, t2, t3, sym_forecast_mul, sym_forecast1, sym_forecast2, sym_forecast3, \
                                                                        = MYclass.Forecast_SYM_H(prop_class,'both')
                                                                                                                                                

#%%

testa = np.array([np.asarray(tm),np.asarray(sym_forecast_mul)])

#%%

np.save('multi_sym_forecastnpy',np.array([tm,sym_forecast_mul]))
np.save('ace_sym_forecastnpy',np.array([t1,sym_forecast1]))
np.save('dscovr_sym_forecastnpy',np.array([t2,sym_forecast2]))
np.save('wind_sym_forecastnpy',np.array([t3,sym_forecast3]))

#%% Test forecasts

fig,ax = plt.subplots(1,1,figsize=(6,4),dpi=300)

style.use('ggplot')

color_multi = '#D3D3D3'  # Grey
color_ace = '#1f77b4'    # Blue
color_dscovr = '#2ca02c' # Green
color_wind = '#d62728'   # Red

ax.plot(tm, sym_forecast_mul, label='Multi', alpha=0.4, color=color_multi)
ax.plot(t1, sym_forecast1, label='ACE', alpha=0.4, color=color_ace)
ax.plot(t2, sym_forecast2, label='DSCOVR', alpha=0.4, color=color_dscovr)
ax.plot(t3, sym_forecast3, label='Wind', alpha=0.4, color=color_wind)


ax.tick_params(axis='both',labelsize = 12, direction='out',top = True, right = True, which='both')
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())

ax.set_xlabel('Time',fontsize = 16)
ax.set_ylabel('SYM/H [nT]',fontsize = 16)

ax.legend()

#%% 

sym_real = MYclass.GetSYMdata()['SYM/H, nT']
treal = MYclass.GetSYMdata()['Time']

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


#%% Multi-spacecraft vs Real

# Form 2d lists for time series' + data
mul_list = [tm,sym_forecast_mul]
real_list = [treal,sym_real]

common_time, sym_forecast_mulA, sym_realA = align_and_interpolate_datasets(mul_list,real_list,len(sym_real))
sym_forecast_mulA, sym_realA = sym_forecast_mulA[0], sym_realA[0]

# Before doing cross correlation let's plot these to test with the common time series

plt.plot(common_time, sym_forecast_mulA)
plt.plot(common_time, sym_realA)

#%%

from cross_correlation import cross_correlation

lags = np.arange(-len(common_time) + 1, len(common_time))
dt = 60

# Calculate corresponding time delays
time_delays = lags * dt

time_delays,cross_corr_values = cross_correlation(sym_forecast_mulA, sym_realA, time_delays)

peak_index = np.argmax(np.abs(cross_corr_values))

# Plot the cross-correlation values
plt.figure(figsize=(8, 5))
plt.plot(time_delays/60, cross_corr_values, label='Cross-correlation')
plt.axvline(time_delays[peak_index]/60, color='red', linestyle='--', label='Peak Correlation')
plt.xlabel('Time Delay (minutes)', fontsize=12)
plt.ylabel('Cross-Correlation', fontsize=12)
plt.title('Cross-Correlation DSCOVR vs Wind Forecasts', fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
#plt.savefig('cross_corr_dscovr_wind.png', dpi=300)
plt.show()

print('\nPeak cross-correlation between the multi-spacecraft prediction & real SYM/H is observed at a time delay of', 
      int(time_delays[peak_index]/60), 'minutes')

# Find the index where time_delays is 0
zero_delay_index = np.where(time_delays == 0)[0]

# Print the cross-correlation at time delay = 0
print(f'Cross-correlation at 0 time delay: {cross_corr_values[zero_delay_index][0]}')
print(f'Max cross-correlation: {max(cross_corr_values)}')

#%% ACE vs DSCOVR

# Form 2d lists for time series' + data
ace_list = [t1,sym_forecast1]
dsc_list = [t2,sym_forecast2]

common_time, sym_forecast1A, sym_forecast2A = align_and_interpolate_datasets(ace_list,dsc_list,len(sym_real))
sym_forecast1A, sym_forecast2A = sym_forecast1A[0], sym_forecast2A[0]

# Before doing cross correlation let's plot these to test with the common time series

plt.plot(common_time, sym_forecast1A)
plt.plot(common_time, sym_forecast2A)

#%%

from cross_correlation import cross_correlation

time_delays,cross_corr_values = cross_correlation(sym_forecast1A, sym_forecast2A, common_time)

peak_index = np.argmax(np.abs(cross_corr_values))

# Plot the cross-correlation values
plt.figure(figsize=(8, 5))
plt.plot(time_delays/60, cross_corr_values, label='Cross-correlation')
plt.axvline(time_delays[peak_index]/60, color='red', linestyle='--', label='Peak Correlation')
plt.xlabel('Time Delay (minutes)', fontsize=12)
plt.ylabel('Cross-Correlation', fontsize=12)
plt.title('Cross-Correlation DSCOVR vs Wind Forecasts', fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
#plt.savefig('cross_corr_dscovr_wind.png', dpi=300)
plt.show()

print('\nPeak cross-correlation between the multi-spacecraft prediction & real SYM/H is observed at a time delay of', 
      int(time_delays[peak_index]/60), 'minutes')

# Find the index where time_delays is 0
zero_delay_index = np.where(time_delays == 0)[0]

# Print the cross-correlation at time delay = 0
print(f'Cross-correlation at 0 time delay: {cross_corr_values[zero_delay_index][0]}')
print(f'Max cross-correlation: {max(cross_corr_values)}')


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
#time_series = time_series.reset_index(drop=True)

#%%

time_min_mul = t1[sym_forecast_mul.index(min(sym_forecast_mul))]
time_min_sym = treal[sym_real.index(min(sym_real))]

#%% Okay, so peak difference around 30 mins which is ~accurate according to Burton et al.
### So let's try scipy's CC method instead

cc_test = scipy.signal.correlate(sym_forecast_mulA, sym_realA, mode='full', method='auto')

lags = np.arange(-len(common_time) + 1, len(common_time))
dt = 60
time_delays = lags * dt

plt.plot(time_delays,cc_test)

peak_index = np.argmax(np.abs(cc_test))

print('\nPeak cross-correlation between the multi-spacecraft prediction & real SYM/H is observed at a time delay of', 
      int(time_delays[peak_index]/60), 'minutes')


#%% Check spacecraft separations over range we've isolated

ad_seplist = df_ACEfilt['x'].values - df_DSCfilt['Wind, Xgse,Re'].values

plt.plot(pd.to_datetime(df_ACEfilt['Time'],unit='s'),ad_seplist)
date_format = mdates.DateFormatter('%d-%m')
plt.gca().xaxis.set_major_formatter(date_format)



