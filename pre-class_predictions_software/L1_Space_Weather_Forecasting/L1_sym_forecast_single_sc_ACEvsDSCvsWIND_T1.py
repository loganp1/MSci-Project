# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:43:46 2024

@author: logan
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from forecasting_model import SYM_forecast
from scipy.optimize import curve_fit

#%%

# Import actual SYM/H data

df_sym = pd.read_csv('SYM1_unix.csv')

df_sym['DateTime'] = pd.to_datetime(df_sym['Time'], unit='s')

df_sym = df_sym[541280:] # Do this as only acceptable data for wind in T1 is past here

DateTime = df_sym['DateTime'].values
sym = df_sym['SYM/H, nT'].values

plt.plot(DateTime,sym)

#%%

##################################################### SPACECRAFT 1 #################################################
####################################################################################################################

# Read in B field and plasma data for the DSCOVR dataset
df_params1 = pd.read_csv('dscovr_T1_1min_unix.csv')
df_params1 = df_params1[541280:]


#%% Clean Data

# Identify rows with NaN values from USEFUL columns
df_params1['BZ, GSE, nT'] = df_params1['BZ, GSE, nT'].replace(9999.990, np.nan)
df_params1['Speed, km/s'] = df_params1['Speed, km/s'].replace(99999.900, np.nan)
df_params1['Proton Density, n/cc'] = df_params1['Proton Density, n/cc'].replace(999.999, np.nan)

# Check where values will be interpolated
Bz1 = df_params1['BZ, GSE, nT'].values
plt.plot(DateTime,Bz1)

#%%

# Interpolate NaN values
for column in df_params1.columns:
    df_params1[column] = df_params1[column].interpolate()

# Drop rows with remaining NaN values
df_params1 = df_params1.dropna()

# Isolate the different USEFUL columns
Bz1 = df_params1['BZ, GSE, nT'].values
vtot1 = df_params1['Speed, km/s'].values
pdens1 = df_params1['Proton Density, n/cc'].values

# Test plotting Bz1 again with interpolation
plt.plot(DateTime,Bz1)


#%% Extract quantities needed for SYM/H prediction and assign functions

def P(proton_density,velocity_mag):
    # Found the correct unit scaling on OMNIweb using units data is given in!
    return proton_density*velocity_mag**2*2e-6

def E(velocity_mag,Bz):
    
    return -velocity_mag*Bz*1e-3

P1 = P(pdens1,vtot1)
E1 = E(vtot1,Bz1)

# Plot E and P to observe distributions

plt.plot(DateTime,P1,label='P')
plt.plot(DateTime,E1,label='E')
plt.xlabel('Day')
plt.ylabel('E (mV/m) & P (nPa)')
plt.grid()

plt.legend()


#%% Use model to forecast SYM/H, starting with a basic model without propagation

# Extract initial SYM/H value for model

sym_forecast1 = []
initial_sym = sym[0]  # Record this for later, as current_sym will change
current_sym = sym[0]

for i in range(len(DateTime)-1):
    
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
plt.plot(DateTime, sym_forecast1, label = 'DSCOVR Forecasted SYM/H')

plt.legend(loc='upper left',fontsize=15)
plt.show()




##################################################### SPACECRAFT 2 #################################################
#%% ################################################################################################################


# Read in B field and plasma data for the WIND dataset
df_params2 = pd.read_csv('wind_T1_1min_unix.csv')
df_params2 = df_params2[541280:]

#%% Clean Data

# Identify rows with NaN values from USEFUL columns
df_params2['BZ, GSE, nT'] = df_params2['BZ, GSE, nT'].replace(9999.990, np.nan)
df_params2['KP_Speed, km/s'] = df_params2['KP_Speed, km/s'].replace(99999.900, np.nan)
df_params2['Kp_proton Density, n/cc'] = df_params2['Kp_proton Density, n/cc'].replace(999.99, np.nan)

# Check where values will be interpolated
Bz2 = df_params2['BZ, GSE, nT'].values
plt.plot(DateTime, Bz2)

#%%

# Interpolate NaN values
for column in df_params2.columns:
    df_params2[column] = df_params2[column].interpolate()

# Drop rows with remaining NaN values
df_params2 = df_params2.dropna()

# Isolate the different USEFUL columns
Bz2 = df_params2['BZ, GSE, nT'].values
vtot2 = df_params2['KP_Speed, km/s'].values
pdens2 = df_params2['Kp_proton Density, n/cc'].values

# Test plotting Bz2 again with interpolation
plt.plot(DateTime, Bz2)
plt.show()
plt.plot(DateTime, vtot2)
plt.show()
plt.plot(DateTime, pdens2)
plt.show()

#%% Extract quantities needed for SYM/H prediction and apply functions

P2 = P(pdens2, vtot2)
E2 = E(vtot2, Bz2)

# Plot E and P to observe distributions

plt.plot(DateTime, P2, label='P')
plt.plot(DateTime, E2, label='E')
plt.xlabel('Day')
plt.ylabel('E (mV/m) & P (nPa)')
plt.grid()

plt.legend()

#%% Use model to forecast SYM/H, starting with a basic model without propagation

# Extract initial SYM/H value for model

sym_forecast2 = []
initial_sym = sym[0]  # Record this for later, as current_sym will change
current_sym = sym[0]

for i in range(len(DateTime)-1):
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
plt.title('DSCOVR Prediction', fontsize=15)

# Set x-axis ticks to display only whole numbers
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Plot forecasted SYM/H in storm to compare to calculated SYM/H
plt.plot(DateTime, sym_forecast2, label='DSCOVR Forecasted SYM/H')

plt.legend(loc='upper left', fontsize=15)
# plt.savefig(path)
plt.show()





##################################################### SPACECRAFT 3 #################################################
#%% ################################################################################################################

# Read in B field and plasma data for the WIND dataset
df_params3 = pd.read_csv('ace_T1_1min_unix.csv')
df_params3 = df_params3[541280:]

#%% Data already cleaned for ACE from SPEDAS

# Check where values will be interpolated
Bz3 = df_params3['Bz'].values
plt.plot(DateTime, Bz3)

#%%

# Interpolate NaN values
for column in df_params3.columns:
    df_params3[column] = df_params3[column].interpolate()

# Drop rows with remaining NaN values
df_params3 = df_params3.dropna()

# Isolate the different USEFUL columns
Bz3 = df_params3['Bz'].values
vx = df_params3['vx'].values
vy = df_params3['vy'].values
vz = df_params3['vz'].values
vtot3 = np.sqrt(vx**2+vy**2+vz**2)
pdens3 = df_params3['n'].values

# Test plotting Bz3 again with interpolation
plt.plot(DateTime, Bz3)
plt.show()
plt.plot(DateTime, vtot3)
plt.show()
plt.plot(DateTime, pdens3)
plt.show()

#%% Extract quantities needed for SYM/H prediction and apply functions

P3 = P(pdens3, vtot3)
E3 = E(vtot3, Bz3)

# Plot E and P to observe distributions

plt.plot(DateTime, P3, label='P')
plt.plot(DateTime, E3, label='E')
plt.xlabel('Day')
plt.ylabel('E (mV/m) & P (nPa)')
plt.grid()

plt.legend()

#%% Use model to forecast SYM/H, starting with a basic model without propagation

# Extract initial SYM/H value for model

sym_forecast3 = []
initial_sym = sym[0]  # Record this for later, as current_sym will change
current_sym = sym[0]

for i in range(len(DateTime)-1):
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
plt.title('DSCOVR Prediction', fontsize=15)

# Set x-axis ticks to display only whole numbers
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Plot forecasted SYM/H in storm to compare to calculated SYM/H
plt.plot(DateTime, sym_forecast3, label='DSCOVR Forecasted SYM/H')

# path = ('C:\\Users\\logan\\OneDrive - Imperial College London\\Uni\\Year 4\\MSci Project\\Figures\\'
#        'step1_SYMH_prediction_2001_storm3.png')

plt.legend(loc='upper left', fontsize=15)
# plt.savefig(path)
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
plt.plot(DateTime, sym_forecast3, label='ACE Forecasted SYM/H')
plt.plot(DateTime, sym_forecast1, label='DSCOVR Forecasted SYM/H')
plt.plot(DateTime, sym_forecast2, label='WIND Forecasted SYM/H')

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
plt.savefig('cross_corr_filtered_dscovr_wind.png', dpi=300)
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
