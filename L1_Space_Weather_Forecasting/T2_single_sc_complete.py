# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 20:11:17 2024

@author: logan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from forecasting_model import SYM_forecast
from scipy.optimize import curve_fit
from singleSC_propagator_function import EP_singleSC_propagator

#%%

# Import actual SYM/H data

df_sym = pd.read_csv('SYM2_unix.csv')

df_sym['DateTime'] = pd.to_datetime(df_sym['Time'], unit='s')

DateTime = df_sym['DateTime'].values
sym = df_sym['SYM/H, nT'].values

plt.plot(DateTime,sym)

#%%

##################################################### SPACECRAFT 1 #################################################
####################################################################################################################

# Read in B field and plasma data for the DSCOVR dataset
df_params1 = pd.read_csv('dscovr_T2_1min_unix.csv')

#%% In this new file, I'll use my single_sc function to return the time series and E,P values downstream

time1, E1, P1, dTs = EP_singleSC_propagator(df_params1,'DSCOVR')
DateTime1 = pd.to_datetime(time1, unit='s')

#%% Use model to forecast SYM/H, starting with a basic model without propagation

# Extract initial SYM/H value for model

sym_forecast1 = []
initial_sym = sym[0]  # Record this for later, as current_sym will change
current_sym = sym[0]

for i in range(len(DateTime1)-1):
    
    new_sym = SYM_forecast(current_sym,
                                     P1[i],
                                     P1[i+1],
                                     E1[i])
    sym_forecast1.append(new_sym)
    current_sym = new_sym
    

sym_forecast1.insert(0,initial_sym)   # Add initial value we used to propagate through forecast

#%% Plot results

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(DateTime, sym, label = 'Measured SYM/H')
plt.grid()

# Adding labels and title
plt.xlabel('Day', fontsize=15)
plt.ylabel('SYM/H (nT)', fontsize=15)
plt.title('DSCOVR Prediction', fontsize=15)

# Set x-axis ticks to display only whole numbers
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Plot forecasted SYM/H in storm to compare to calculated SYM/H
plt.plot(DateTime1, sym_forecast1, label = 'DSCOVR Forecasted SYM/H')

plt.legend(loc='upper left',fontsize=15)
plt.show()


#%% Plot time shift histogram

# Set a custom color and edgecolor for the bars
plt.hist(dTs/60, bins=100, color='#3498db', edgecolor='black', alpha=0.7)

# Add a title to the histogram
plt.title('Distribution of Time Shifts')

# Label the x and y axes
plt.xlabel('Time Shift (minutes)')
plt.ylabel('Frequency')

# Add grid lines to improve readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust the x-axis ticks for better readability
plt.xticks(range(int(min(dTs/60)), int(max(dTs/60))+1, 5))

# Add a vertical line at the mean or any other relevant metric
mean_value = np.mean(dTs/60)
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.2f} mins')

# Show legend
plt.legend()

# Display the plot
plt.show()



##################################################### SPACECRAFT 2 #################################################
#%% ################################################################################################################


# Read in B field and plasma data for the WIND dataset
df_params2 = pd.read_csv('wind_T2_1min_unix.csv')

#%% In this new file, I'll use my single_sc function to return the time series and E,P values downstream

time2, E2, P2, dTs2 = EP_singleSC_propagator(df_params2,'Wind')
DateTime2 = pd.to_datetime(time2, unit='s')

#%% Use model to forecast SYM/H, starting with a basic model without propagation

# Extract initial SYM/H value for model

sym_forecast2 = []
initial_sym = sym[0]  # Record this for later, as current_sym will change
current_sym = sym[0]

for i in range(len(DateTime2)-1):
    new_sym = SYM_forecast(current_sym, P2[i], P2[i+1], E2[i])
    sym_forecast2.append(new_sym)
    current_sym = new_sym

sym_forecast2.insert(0, initial_sym)   # Add initial value we used to propagate through forecast

#%% Plot results

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(DateTime, sym, label='Measured SYM/H')
plt.grid()

# Adding labels and title
plt.xlabel('Day', fontsize=15)
plt.ylabel('SYM/H (nT)', fontsize=15)
plt.title('Wind Prediction', fontsize=15)

# Set x-axis ticks to display only whole numbers
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Plot forecasted SYM/H in storm to compare to calculated SYM/H
plt.plot(DateTime2, sym_forecast2, label='Wind Forecasted SYM/H')

plt.legend(loc='upper left', fontsize=15)
# plt.savefig(path)
plt.show()

#%% Plot time shift histogram

# Set a custom color and edgecolor for the bars
plt.hist(dTs2/60, bins=100, color='#3498db', edgecolor='black', alpha=0.7)

# Add a title to the histogram
plt.title('Distribution of Time Shifts')

# Label the x and y axes
plt.xlabel('Time Shift (minutes)')
plt.ylabel('Frequency')

# Add grid lines to improve readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust the x-axis ticks for better readability
plt.xticks(range(int(min(dTs2/60)), int(max(dTs2/60))+1, 5))

# Add a vertical line at the mean or any other relevant metric
mean_value = np.mean(dTs2/60)
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.2f} mins')

# Show legend
plt.legend()

# Display the plot
plt.show()



##################################################### SPACECRAFT 3 #################################################
#%% ################################################################################################################

# Read in B field and plasma data for the WIND dataset
df_params3 = pd.read_csv('ace_T2_1min_unix.csv')

#%% In this new file, I'll use my single_sc function to return the time series and E,P values downstream

time3, E3, P3, dTs3 = EP_singleSC_propagator(df_params3,'ACE')
DateTime3 = pd.to_datetime(time3, unit='s')

#%% Use model to forecast SYM/H, starting with a basic model without propagation

# Extract initial SYM/H value for model

sym_forecast3 = []
initial_sym = sym[0]  # Record this for later, as current_sym will change
current_sym = sym[0]

for i in range(len(DateTime3)-1):
    new_sym = SYM_forecast(current_sym, P3[i], P3[i+1], E3[i])
    sym_forecast3.append(new_sym)
    current_sym = new_sym

sym_forecast3.insert(0, initial_sym)   # Add initial value we used to propagate through forecast

#%% Plot results

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(DateTime, sym, label='Measured SYM/H')
plt.grid()

# Adding labels and title
plt.xlabel('Day', fontsize=15)
plt.ylabel('SYM/H (nT)', fontsize=15)
plt.title('ACE Prediction', fontsize=15)

# Set x-axis ticks to display only whole numbers
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Plot forecasted SYM/H in storm to compare to calculated SYM/H
plt.plot(DateTime3, sym_forecast3, label='ACE Forecasted SYM/H')

# path = ('C:\\Users\\logan\\OneDrive - Imperial College London\\Uni\\Year 4\\MSci Project\\Figures\\'
#        'step1_SYMH_prediction_2001_storm3.png')

plt.legend(loc='upper left', fontsize=15)
# plt.savefig(path)
plt.show()


#%% Plot time shift histogram

# Set a custom color and edgecolor for the bars
plt.hist(dTs3/60, bins=100, color='#3498db', edgecolor='black', alpha=0.7)

# Add a title to the histogram
plt.title('Distribution of Time Shifts')

# Label the x and y axes
plt.xlabel('Time Shift (minutes)')
plt.ylabel('Frequency')

# Add grid lines to improve readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust the x-axis ticks for better readability
plt.xticks(range(int(min(dTs3/60)), int(max(dTs3/60))+1, 5))

# Add a vertical line at the mean or any other relevant metric
mean_value = np.mean(dTs3/60)
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.2f} mins')

# Show legend
plt.legend()

# Display the plot
plt.show()



############################################# Now compare all 3 spacecraft #########################################
#%% ################################################################################################################

plt.figure(figsize=(12, 6))
plt.grid()

# Adding labels and title
plt.xlabel('Day', fontsize=15)
plt.ylabel('SYM/H (nT)', fontsize=15)

# Set x-axis ticks
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Plot forecasted SYM/H in storm to compare to calculated SYM/H
plt.plot(DateTime3, sym_forecast3, label='ACE Forecasted SYM/H')
plt.plot(DateTime1, sym_forecast1, label='DSCOVR Forecasted SYM/H')
plt.plot(DateTime2, sym_forecast2, label='WIND Forecasted SYM/H')

plt.legend(fontsize=15)

plt.show()


#%% Plot small range to observe differences

downtoval = 100000
uptoval = 150000

plt.figure(figsize=(12, 6))
plt.grid()

# Adding labels and title
plt.xlabel('Date', fontsize=15)
plt.ylabel('SYM/H (nT)', fontsize=15)

# Set x-axis ticks
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Plot forecasted SYM/H in storm to compare to calculated SYM/H
plt.plot(DateTime1[downtoval:uptoval], sym_forecast3[downtoval:uptoval], label='ACE Forecasted SYM/H')
plt.plot(DateTime2[downtoval:uptoval], sym_forecast1[downtoval:uptoval], label='DSCOVR Forecasted SYM/H')
plt.plot(DateTime3[downtoval:uptoval], sym_forecast2[downtoval:uptoval], label='WIND Forecasted SYM/H')
plt.plot(DateTime[downtoval:uptoval], sym[downtoval:uptoval], label='Measured SYM/H')

plt.legend(fontsize=15)

plt.show()


#%% Perform cross-correlation analysis 

from cross_correlation import cross_correlation

lags = np.arange(-len(sym_forecast1) + 1, len(sym_forecast1))
dt = 60

# Calculate corresponding time delays
time_delays = lags * dt

time_delays12,cross_corr_values12 = cross_correlation(sym_forecast1, sym_forecast2,time_delays)
time_delays13,cross_corr_values13 = cross_correlation(sym_forecast1, sym_forecast3,time_delays)
time_delays23,cross_corr_values23 = cross_correlation(sym_forecast2, sym_forecast3,time_delays)

# Find the index of the maximum cross-correlation value
peak_index12 = np.argmax(np.abs(cross_corr_values12))
peak_index13 = np.argmax(np.abs(cross_corr_values13))
peak_index23 = np.argmax(np.abs(cross_corr_values23))


# Plot the cross-correlation values 12
plt.figure(figsize=(8, 5))
plt.plot(time_delays12/60, cross_corr_values12, label='Cross-correlation')
plt.axvline(time_delays12[peak_index12]/60, color='red', linestyle='--', label='Peak Correlation')
plt.xlabel('Time Delay (minutes)', fontsize=12)
plt.ylabel('Cross-Correlation', fontsize=12)
plt.title('Cross-Correlation DSCOVR vs Wind Forecasts', fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
#plt.savefig('cross_corr_dscovr_wind.png', dpi=300)
plt.show()

print('\nPeak cross-correlation between DSCOVR and Wind is observed at a time delay of', 
      int(time_delays[peak_index12]/60), 'minutes')

# Repeat the same enhancements for the other plots

# Plot the cross-correlation values 13
plt.figure(figsize=(8, 5))
plt.plot(time_delays13/60, cross_corr_values13, label='Cross-correlation')
plt.axvline(time_delays13[peak_index13]/60, color='red', linestyle='--', label='Peak Correlation')
plt.xlabel('Time Delay (minutes)', fontsize=12)
plt.ylabel('Cross-Correlation', fontsize=12)
plt.title('Cross-Correlation DSCOVR vs ACE Forecasts', fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
#plt.savefig('cross_corr_dscovr_ace.png', dpi=300)
plt.show()

print('\nPeak cross-correlation between DSCOVR and ACE is observed at a time delay of', 
      int(time_delays[peak_index13]/60), 'minutes')

# Plot the cross-correlation values 23
plt.figure(figsize=(8, 5))
plt.plot(time_delays23/60, cross_corr_values23, label='Cross-correlation')
plt.axvline(time_delays23[peak_index23]/60, color='red', linestyle='--', label='Peak Correlation')
plt.xlabel('Time Delay (minutes)', fontsize=12)
plt.ylabel('Cross-Correlation', fontsize=12)
plt.title('Cross-Correlation Wind vs ACE Forecasts', fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
#plt.savefig('cross_corr_wind_ace.png', dpi=300)
plt.show()

print('\nPeak cross-correlation between Wind and ACE is observed at a time delay of', 
      int(time_delays[peak_index23]/60), 'minutes')


#%% Fit Gaussians around the peaks

def Gaussian(x, A, mu, sigma):
    
    return A * np.exp(-(x - mu)**2/(2 * sigma**2))

# Isolate the peak of the cross-correlation curve
lower_lim = -100*60   # x60 because graph shows in minutes but data is in seconds, so convert mins-->secs
upper_lim = 50*60

# Ensure that the boolean array matches the size of the original arrays
boolean_mask = (time_delays >= lower_lim) & (time_delays <= upper_lim)

# Filter the arrays based on the limits
filtered_cross_corr_values12 = cross_corr_values12[boolean_mask]
filtered_time_delays12 = time_delays12[boolean_mask]

# Plot the cross-correlation values for the filtered data
plt.figure(figsize=(8, 5))
plt.plot(filtered_time_delays12/60, filtered_cross_corr_values12, label='Filtered Cross-Correlation')

# Find the index of the peak in the filtered data
peak_index = np.argmax(filtered_cross_corr_values12)

# Plot the peak in the filtered data
plt.axvline(filtered_time_delays12[peak_index]/60, color='red', linestyle='--', label='Peak Correlation')

# Fitting code for the new filtered data
popt, pcov = curve_fit(Gaussian, filtered_time_delays12/60, filtered_cross_corr_values12)
plt.plot(filtered_time_delays12/60, Gaussian(filtered_time_delays12/60, *popt), label='Gaussian Fit')

plt.xlabel('Time Delay (minutes)', fontsize=12)
plt.ylabel('Cross-Correlation', fontsize=12)
plt.title('Cross-Correlation DSCOVR vs Wind', fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
#plt.savefig('cross_corr_filtered_dscovr_wind.png', dpi=300)
plt.show()

print('\nPeak cross-correlation is observed', int(filtered_time_delays12[peak_index]/60), 'minutes after bow shock measurement')
print('\nGaussian fit parameters:')
print('A =', popt[0], '\nmu =', popt[1], '\nsigma =', popt[2])


#%% Plot random test Gaussian with calculated params

A = popt[0]
mu = popt[1]
sigma = popt[2]

X = np.linspace(-1000,1000,1000)
y = Gaussian(X,A,mu,sigma)

plt.plot(X,y)
