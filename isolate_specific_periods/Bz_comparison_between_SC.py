# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:06:08 2024

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

data = pd.read_csv('DSCOVR_data_unix.csv')
data['BZ, GSE, nT'] = data['BZ, GSE, nT'].replace(9999.99, np.nan)
data2=pd.read_csv('ACE_data_unix.csv')
data3=pd.read_csv('Wind_data_unix.csv')
#%%
plt.plot(data['Time'].values[:1000],data['BZ, GSE, nT'].values[:1000])

data['Wind, Xgse,Re'] = data['Wind, Xgse,Re'].replace(9999.99, np.nan)
data3['Wind, Xgse,Re'] = data3['Wind, Xgse,Re'].replace(9999.990000, np.nan)

data['Proton Density, n/cc'] = data['Proton Density, n/cc'].replace(999.999, np.nan)
data3['Kp_proton Density, n/cc'] = data3['Kp_proton Density, n/cc'].replace(999.990000, np.nan)
#%%
# Convert 'Time' column to datetime if it's not already
data['dTime'] = pd.to_datetime(data['Time'], unit='s')

# Specify the time value to filter after
time_threshold = pd.to_datetime(1504810000.0, unit='s')
time_upper = pd.to_datetime(1504882380.0, unit='s')

# Filter data based on the time threshold
filtered_data = data[(data['dTime'] > time_threshold) & (data['dTime'] < time_upper)]

# Plot the filtered data
plt.plot(filtered_data['dTime'].values, filtered_data['BZ, GSE, nT'].values,label='DSC')
plt.xlabel('Time')
plt.ylabel('BZ, GSE, nT')
plt.title('Data Points After 1504877580.0')

# Convert 'Time' column to datetime if it's not already
data2['dTime'] = pd.to_datetime(data2['Time'], unit='s')

# Filter data based on the time threshold
filtered_data2 = data2[(data2['dTime'] > time_threshold) & (data2['dTime'] < time_upper)]

# Plot the filtered data
plt.plot(filtered_data2['dTime'].values, filtered_data2['Bz'].values,label='ACE')

# data3
# Convert 'Time' column to datetime if it's not already
data3['dTime'] = pd.to_datetime(data3['Time'], unit='s')

# Filter data based on the time threshold
filtered_data3 = data3[(data3['dTime'] > time_threshold) & (data3['dTime'] < time_upper)]

# Plot the filtered data
plt.plot(filtered_data3['dTime'].values, filtered_data3['BZ, GSE, nT'].values,label='Wind')

plt.legend()
plt.show()

# Find the index of the maximum in each dataset
max_index_data = filtered_data['BZ, GSE, nT'].idxmax()
max_index_data2 = filtered_data2['Bz'].idxmax()

# Extract the corresponding timestamps
timestamp_data = filtered_data.loc[max_index_data, 'dTime']
timestamp_data2 = filtered_data2.loc[max_index_data2, 'dTime']

# Calculate the time separation
time_separation = timestamp_data - timestamp_data2

print(f'Time Separation between Maxima: {time_separation}')

#%%
import matplotlib.dates as mdates

plt.plot(filtered_data['dTime'].values,filtered_data['Wind, Xgse,Re']*6378,label='DSCOVR')
plt.plot(filtered_data2['dTime'].values,filtered_data2['x'],label='ACE')
plt.plot(filtered_data3['dTime'].values,filtered_data3['Wind, Xgse,Re']*6378,label='Wind')
plt.legend()
# Set the x-axis locator to every 6 hours
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))

# Format the x-axis labels to show the time
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))


#%% Try same for n data

# Convert 'Time' column to datetime if it's not already
data['dTime'] = pd.to_datetime(data['Time'], unit='s')

# Specify the time value to filter after
time_threshold = pd.to_datetime(1504800000.0, unit='s')
time_upper = pd.to_datetime(1504842380.0, unit='s')

# Filter data based on the time threshold
filtered_data = data[(data['dTime'] > time_threshold) & (data['dTime'] < time_upper)]

# Plot the filtered data
plt.plot(filtered_data['dTime'].values, filtered_data['Proton Density, n/cc'].values,label='DSC')
plt.xlabel('Time')
plt.ylabel('Proton Density, n/cc')
plt.title('Data Points After 1504877580.0')

# Convert 'Time' column to datetime if it's not already
data2['dTime'] = pd.to_datetime(data2['Time'], unit='s')

# Filter data based on the time threshold
filtered_data2 = data2[(data2['dTime'] > time_threshold) & (data2['dTime'] < time_upper)]

# Plot the filtered data
plt.plot(filtered_data2['dTime'].values, filtered_data2['n'].values,label='ACE')

# data3
# Convert 'Time' column to datetime if it's not already
data3['dTime'] = pd.to_datetime(data3['Time'], unit='s')

# Filter data based on the time threshold
filtered_data3 = data3[(data3['dTime'] > time_threshold) & (data3['dTime'] < time_upper)]

# Plot the filtered data
plt.plot(filtered_data3['dTime'].values, filtered_data3['Kp_proton Density, n/cc'].values,label='Wind')

plt.legend()
plt.show()

# Find the index of the maximum in each dataset
max_index_data = filtered_data['Proton Density, n/cc'].idxmax()
max_index_data2 = filtered_data2['n'].idxmax()

# Extract the corresponding timestamps
timestamp_data = filtered_data.loc[max_index_data, 'dTime']
timestamp_data2 = filtered_data2.loc[max_index_data2, 'dTime']

# Calculate the time separation
time_separation = timestamp_data - timestamp_data2

print(f'Time Separation between Maxima: {time_separation}')


#%% Let's do a CC between ACE and the others to see if there is a characteristic time shift

# Calculate cross-correlation using the formula
cross_corr = np.correlate(data['BZ, GSE, nT'].values, data2['Bz'].values, mode='same') 

# Calculate the time lags corresponding to the cross-correlation values
#time_lags = np.arange(-len(x) + 1, len(x))
time_lags = np.arange(-len(cross_corr)//2, len(cross_corr)//2)

# Calculate delta t for each time lag using the interpolated_time_series
delta_t = data['Time'][1] - data['Time'][0]

#%%

plt.plot(delta_t,cross_corr)