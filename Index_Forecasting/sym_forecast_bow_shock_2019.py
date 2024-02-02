# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:45:09 2024

@author: logan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read in SYM/H index & plasma data
df_params = pd.read_csv('omni_ind_params_2019_1min.csv')


#%% Clean Data

# Identify rows with NaN values
df_params['BZ, nT (GSE)'] = df_params['BZ, nT (GSE)'].replace(9999.99, np.nan)
df_params['Speed, km/s'] = df_params['Speed, km/s'].replace(99999.9, np.nan)
df_params['Flow pressure, nPa'] = df_params['Flow pressure, nPa'].replace(99.99, np.nan)
df_params['Electric field, mV/m'] = df_params['Electric field, mV/m'].replace(999.99, np.nan)
df_params['Proton Density, n/cc'] = df_params['Proton Density, n/cc'].replace(999.9, np.nan)

# Interpolate NaN valuesind_params_2001_1min.csv'
for column in df_params.columns[3:]:  # Exclude the time columns
    df_params[column] = df_params[column].interpolate()

# Drop rows with remaining NaN values
df_params = df_params.dropna()

# Isolate the different columns
sym = df_params['SYM/H, nT']
Bz = df_params['BZ, nT (GSE)']
speed = df_params['Speed, km/s']
pressure = df_params['Flow pressure, nPa']
Efield = df_params['Electric field, mV/m']
density = df_params['Proton Density, n/cc']

#%% Forecasting Function

# Create injection energy function of E
def F(E, d):
    
    if E < 0.5:
        F = 0
    else:
        F = d * (E - 0.5)
    return F


# Record different terms 
deriv_term = []
F_E = []
term1 = []
overall_contr = []

# Model from Burton et al. 1975
def SYM_forecast(SYM_i, dt, a, b, c, d, P_i, P_iplus1, E_i):
    
    derivative_term = b * (P_iplus1**0.5 - P_i**0.5)/dt
    deriv_term.append(derivative_term*dt)
    F_E.append(F(E_i,d)*dt)
    term1.append(-a*(SYM_i - b * np.sqrt(P_i) + c)*dt)
    overall_contr.append((-a * (SYM_i - b * np.sqrt(P_i) + c) + F(E_i, d) + derivative_term) * dt)
    
    return SYM_i + (-a * (SYM_i - b * np.sqrt(P_i) + c) + F(E_i, d) + derivative_term) * dt


#%% Plot E and P distributions for visual comparison vs SYM/H

# Create plot of storm1 in 2001 SYM/H data

# Combine 'Day' and 'Hour' into a new column 'DayHour'
df_params['DayHourMin'] = df_params['Day'] + df_params['Hour'] / 24.0 + df_params['Minute'] / 60 / 24

# Filter the DataFrame
df_storm1 = df_params[(132 < df_params['Day']) & (df_params['Day'] < 136)]

# Extract relevant columns
days1 = df_storm1['DayHourMin'].values
sym_storm1 = df_storm1['SYM/H, nT'].values
Efield = df_storm1['Electric field, mV/m'].values
pressure = df_storm1['Flow pressure, nPa'].values

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(days1, sym_storm1/-15, label = 'SYM/H')
plt.grid()

# Adding labels and title
plt.xlabel('Day in 2001', fontsize=15)
plt.ylabel('Scaled SYM/H (nT) (/-15)', fontsize=15)
#plt.title('Storm 1', fontsize=15)

# Set x-axis ticks to display only whole numbers
plt.xticks(range(int(days1[0]), int(days1[-1]) + 1), fontsize=15)
plt.yticks(fontsize=15)

# Plot parameters E and P in storm to compare to SYM/H
plt.plot(days1, Efield, label = 'E')
plt.plot(days1, pressure, label = 'P')

plt.legend()

path = ('C:\\Users\\logan\\OneDrive - Imperial College London\\Uni\\Year 4\\MSci Project\\Figures\\'
        'step1_SYMH_E_P_data_2001_storm1.png')

plt.savefig(path)


#%% Forecast method: 
    
    # Take a SYM/H measurement at time t, given by start of stormi period of data as previously used
    # Have measurements for the parameters E = vBz, P = nv as propagated to the bow shock from L1
    # Need to know a time for the impact on the index from bow shock to Earth to know which params = which time
    
# Add constants

dt = 60 # seconds in an minute, as one timestep is one minute for SYM/H

# Unit gamma is used (1nT)
gamma = 1 # BECAUSE OUR DATAFRAMES HAVE UNITS OF nT!!!

# Set your parameter values (a, b, c)
a = 3.6e-5
b = 0.2 * gamma
c = 0 * gamma
d = -1.5e-3 * gamma

#%% Storm1 Forecast
    
# Initially, let's act under the assumption of no time delay from bow shock to Earth

# Filter the DataFrame
#df_storm1 = df_params[(132 < df_params['Day']) & (df_params['Day'] < 136)]

sym_forecast_storm1 = []
current_sym = df_storm1['SYM/H, nT'].tolist()[0]

# Reset the arrays for the different terms in the model
deriv_term = []
F_E = []
term1 = []
overall_contr = []

for i in range(len(df_storm1['Day'].tolist())-1):
    
    new_sym = SYM_forecast(current_sym,dt,a,b,c,d,
                                     df_storm1['Flow pressure, nPa'].tolist()[i],
                                     df_storm1['Flow pressure, nPa'].tolist()[i+1],
                                     df_storm1['Electric field, mV/m'].tolist()[i])
    sym_forecast_storm1.append(new_sym)
    current_sym = new_sym
    
    

#%% Plot the contributions from the different terms

plt.figure()
plt.plot(days1[:-1],deriv_term,label='Derivative Term')
plt.plot(days1[:-1],F_E,label='F(E) Term')
plt.plot(days1[:-1],term1,label='1st Term')
#plt.plot(days1[:-1],overall_contr,label='Overall Contribution')
plt.legend()
plt.show()

# Plot cumulative sum of overall_contr which should show how wed expect sym forecast to change
plt.figure()
plt.plot(days1[:-1],np.cumsum(overall_contr))    
plt.xlabel('Day in 2001', fontsize=10)
plt.ylabel('Cumulative Sum of Terms in Model', fontsize=10)
plt.grid()


#%% Plot forecast vs calculated SYM/H data

# Storm1 Plot

# Extract relevant columns
days1 = df_storm1['DayHourMin'].values             # .values just the Pandas version of .tolist I think
sym_storm1 = df_storm1['SYM/H, nT'].values

sym_forecast_storm1[0] = sym_storm1[0]   # Add initial valu we use to propagate through forecast so
                                         # arrays are same length, ready for cross-correlation analysis

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(days1, sym_storm1, label = 'SYM/H')
plt.grid()

# Adding labels and title
plt.xlabel('Day in 2001', fontsize=15)
plt.ylabel('SYM/H (nT)', fontsize=15)
#plt.title('Storm 1', fontsize=15)

# Set x-axis ticks to display only whole numbers
#plt.xticks([int(day) for day in days1] + [81], fontsize=15)        # very slow so try below
plt.xticks(range(int(days1[0]), int(days1[-1]) + 1), fontsize=15)
plt.yticks(fontsize=15)

# Plot forecasted SYM/H in storm to compare to calculated SYM/H
plt.plot(days1[1:], sym_forecast_storm1, label = 'Forecasted SYM/H')

path = ('C:\\Users\\logan\\OneDrive - Imperial College London\\Uni\\Year 4\\MSci Project\\Figures\\'
        'step1_SYMH_prediction_2001_storm1.png')

plt.legend()
plt.savefig(path)
plt.show()


#%%% Compare forecast to each term in model

plt.figure()
plt.plot(days1[1:],np.asarray(sym_forecast_storm1),label='Scaled SYM/H Prediction')
plt.plot(days1[1:],deriv_term,label='Derivative Term')
plt.plot(days1[1:],np.asarray(F_E)*200,label='F(E) Term')
plt.plot(days1[1:],np.asarray(term1)*-300,label='1st Term')
plt.legend()
plt.show()


#%% Cross-correlations to find a time shift from bow shock to Earth

# Storm1

from scipy.signal import correlate

# Perform cross-correlation
lags = np.arange(-len(sym_storm1) + 1, len(sym_storm1))
#cross_corr_values = correlate(sym_storm1, sym_forecast_storm1, mode='full') / len(sym_storm1)
# Normalise:
#cross_corr_values = correlate(sym_storm1, sym_forecast_storm1, 
#                              mode='full') / (np.std(sym_storm1) * np.std(sym_forecast_storm1) * len(sym_storm1))

# Calculate corresponding time delays
time_delays = lags * dt

# Using my crosscorr function I created
from cross_correlation import cross_correlation
time_delays,cross_corr_values = cross_correlation(sym_storm1, sym_forecast_storm1,time_delays)

# Find the index of the maximum cross-correlation value
peak_index = np.argmax(np.abs(cross_corr_values))

# Plot the cross-correlation values
plt.figure(figsize=(12, 6))
plt.plot(time_delays[:-1]/60, cross_corr_values, label='Cross-Correlation')
plt.axvline(time_delays[peak_index]/60, color='red', linestyle='--', label='Peak Correlation')
plt.xlabel('Time Delay (minutes)', fontsize=15)
plt.ylabel('Cross-Correlation', fontsize=15)
plt.title('Storm1')
plt.grid(True)
#plt.text(-1000,7200,'Peak Cross-Correlation is Observed at \n 65 Minutes After Bow Shock Measurements',
#         horizontalalignment='right')


# Fit a Gaussian to the top (or all?) of the curve to try and quantify a time shift & its uncertainty

from scipy.optimize import curve_fit

def Gaussian(x, A, mu, sigma):
    
    return A * np.exp(-(x - mu)**2/(2 * sigma**2))

popt, pcov = curve_fit(Gaussian, time_delays[:-1]/60, cross_corr_values)

popt, pcov = curve_fit(Gaussian, time_delays[:-1]/60, cross_corr_values)

plt.plot(time_delays[:-1]/60, Gaussian(time_delays[:-1]/60, *popt), label='Gaussian Fit')

plt.legend()
plt.show()

print('\nPeak cross-correlation is observed', int(time_delays[peak_index]/60), 'minutes after bow shock measurement')
print('\nGaussian fit parameters:')
print('A =',popt[0],'\nmu =',popt[1],'\nsigma =',popt[2])


#%% As Tim said, it's meaningless to fit Gaussians over the whole distribution.
#   We are interested in the PEAK

# Isolate the peak of the cross-correlation curve
lower_lim = -70*60   # x60 because graph shows in minutes but data is in seconds, so convert mins-->secs
upper_lim = 100*60

# Ensure that the boolean array matches the size of the original arrays
boolean_mask = (time_delays[:-1] >= lower_lim) & (time_delays[:-1] <= upper_lim)

# Filter the arrays based on the limits
filtered_cross_corr_values = cross_corr_values[boolean_mask]
filtered_time_delays = time_delays[:-1][boolean_mask]

# Plot the cross-correlation values for the filtered data
plt.figure(figsize=(12, 6))
plt.plot(filtered_time_delays/60, filtered_cross_corr_values, label='Filtered Cross-Correlation')

# Find the index of the peak in the filtered data
peak_index = np.argmax(filtered_cross_corr_values)

# Plot the peak in the filtered data
plt.axvline(filtered_time_delays[peak_index]/60, color='red', linestyle='--', label='Peak Correlation')

# Fitting code for the new filtered data
popt, pcov = curve_fit(Gaussian, filtered_time_delays/60, filtered_cross_corr_values)
plt.plot(filtered_time_delays/60, Gaussian(filtered_time_delays/60, *popt), label='Gaussian Fit')

plt.xlabel('Time Delay (minutes)', fontsize=15)
plt.ylabel('Cross-Correlation', fontsize=15)
plt.title('Storm1')
plt.grid(True)
plt.legend()
plt.show()

print('\nPeak cross-correlation is observed', int(filtered_time_delays[peak_index]/60), 'minutes after bow shock measurement')
print('\nGaussian fit parameters:')
print('A =', popt[0], '\nmu =', popt[1], '\nsigma =', popt[2])