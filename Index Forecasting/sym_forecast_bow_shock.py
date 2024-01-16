# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:10:22 2023

@author: logan
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read in SYM/H index & plasma data
df_params = pd.read_csv('C:\\Users\\logan\\VSCode_copy_OMNIWeb_data_official\\omni_ind_params_2001_1min.csv')

# Isolate the different columns
sym = df_params['SYM/H, nT']
Bz = df_params['BZ, nT (GSE)']
speed = df_params['Speed, km/s']
pressure = df_params['Flow pressure, nPa']
Efield = df_params['Electric field, mV/m']
density = df_params['Proton Density, n/cc']


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

#%% Forecasting Function

# Create injetion energy function of E
def F(E, d):
    
    if E < 0.5:
        F = 0
    else:
        F = d * (E - 0.5)
    return F

# Model from Burton et al. 1975
def SYM_forecast(SYM_i, dt, a, b, c, d, P_i, P_iplus1, E_i):
    
    derivative_term = b * (P_iplus1**0.5 - P_i**0.5)/dt
    #print('SYM_i =',SYM_i)
    #print('deriv term =',derivative_term*dt)
    #print('F(E) =',F(E_i,d)*dt)
    #print('1st term =',-a*(SYM_i - b * np.sqrt(P_i) + c)*dt)
    #print('SYM_i+1 =',SYM_i + (-a * (SYM_i - b * np.sqrt(P_i) + c) + F(E_i, d) + derivative_term)*dt)
    return SYM_i + (-a * (SYM_i - b * np.sqrt(P_i) + c) + F(E_i, d) + derivative_term) * dt

#%% Plot E and P distributions for visual comparison vs SYM/H - do for storm 1

# Create plot of storm1 in 2001 SYM/H data

# Combine 'Day' and 'Hour' into a new column 'DayHour'
df_params['DayHourMin'] = df_params['Day'] + df_params['Hour'] / 24.0 + df_params['Minute'] / 60 / 24

# Filter the DataFrame
df_storm1 = df_params[(76 < df_params['Day']) & (df_params['Day'] < 81)]

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
c = 20 * gamma
d = -1.5e-3 * gamma

#%% Storm1 Forecast
    
# Initially, let's act under the assumption of no time delay from bow shock to Earth

# Filter the DataFrame
df_storm1 = df_params[(76 < df_params['Day']) & (df_params['Day'] < 81)]

sym_forecast_storm1 = []
current_sym = df_storm1['SYM/H, nT'].tolist()[0]

for i in range(len(df_storm1['Day'].tolist())-1):
    
    new_sym = SYM_forecast(current_sym,dt,a,b,c,d,
                                     df_storm1['Flow pressure, nPa'].tolist()[i],
                                     df_storm1['Flow pressure, nPa'].tolist()[i+1],
                                     df_storm1['Electric field, mV/m'].tolist()[i])
    sym_forecast_storm1.append(new_sym)
    current_sym = new_sym
    

#%% Plot forecast vs calculated SYM/H data

# Storm1 Plot

# Extract relevant columns
days1 = df_storm1['DayHourMin'].values             # .values just the NumPy version of .tolist I think
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

#%% Storm2 Forecast
    
# Initially, let's act under the assumption of no time delay from bow shock to Earth

# Filter the DataFrame
df_storm2 = df_params[(88 < df_params['Day']) & (df_params['Day'] < 93)]

sym_forecast_storm2 = []
current_sym = df_storm2['SYM/H, nT'].tolist()[0]

for i in range(len(df_storm2['Day'].tolist())-1):
    
    new_sym = SYM_forecast(current_sym,dt,a,b,c,d,
                                     df_storm2['Flow pressure, nPa'].tolist()[i],
                                     df_storm2['Flow pressure, nPa'].tolist()[i+1],
                                     df_storm2['Electric field, mV/m'].tolist()[i])
    sym_forecast_storm2.append(new_sym)
    current_sym = new_sym
    


#%% Storm2 Plot 

# Extract relevant columns
days2 = df_storm2['DayHourMin'].values
sym_storm2 = df_storm2['SYM/H, nT'].values

sym_forecast_storm2[0] = sym_storm2[0]   # Add initial valu we use to propagate through forecast so
                                         # arrays are same length, ready for cross-correlation analysis

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(days2, sym_storm2, label = 'SYM/H')
plt.grid()

# Adding labels and title
plt.xlabel('Day in 2001', fontsize=15)
plt.ylabel('SYM/H (nT)', fontsize=15)
#plt.title('Storm 2', fontsize=15)

# Set x-axis ticks to display only whole numbers
plt.xticks(range(int(days2[0]), int(days2[-1]) + 1), fontsize=15)
plt.yticks(fontsize=15)

# Plot forecasted SYM/H in storm to compare to calculated SYM/H
plt.plot(days2[1:], sym_forecast_storm2, label = 'Forecasted SYM/H')

path = ('C:\\Users\\logan\\OneDrive - Imperial College London\\Uni\\Year 4\\MSci Project\\Figures\\'
        'step1_SYMH_prediction_2001_storm2.png')

plt.legend()
plt.savefig(path)


#%% Storm3 Forecast
    
# Initially, let's act under the assumption of no time delay from bow shock to Earth

# Filter the DataFrame
df_storm3 = df_params[(100 < df_params['Day']) & (df_params['Day'] < 104)]

sym_forecast_storm3 = []
current_sym = df_storm3['SYM/H, nT'].tolist()[0]

for i in range(len(df_storm3['Day'].tolist())-1):
    
    new_sym = SYM_forecast(current_sym,dt,a,b,c,d,
                                     df_storm3['Flow pressure, nPa'].tolist()[i],
                                     df_storm3['Flow pressure, nPa'].tolist()[i+1],
                                     df_storm3['Electric field, mV/m'].tolist()[i])
    sym_forecast_storm3.append(new_sym)
    current_sym = new_sym
    


#%% Storm3 Plot 

# Extract relevant columns
days3 = df_storm3['DayHourMin'].values
sym_storm3 = df_storm3['SYM/H, nT'].values

sym_forecast_storm3[0] = sym_storm3[0]   # Add initial valu we use to propagate through forecast so
                                         # arrays are same length, ready for cross-correlation analysis

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(days3, sym_storm3, label = 'SYM/H')
plt.grid()

# Adding labels and title
plt.xlabel('Day in 2001', fontsize=15)
plt.ylabel('SYM/H (nT)', fontsize=15)
#plt.title('Storm 3', fontsize=15)

# Set x-axis ticks to display only whole numbers
plt.xticks(range(int(days3[0]), int(days3[-1]) + 1), fontsize=15)
plt.yticks(fontsize=15)

# Plot forecasted SYM/H in storm to compare to calculated SYM/H
plt.plot(days3[1:], sym_forecast_storm3, label = 'Forecasted SYM/H')

path = ('C:\\Users\\logan\\OneDrive - Imperial College London\\Uni\\Year 4\\MSci Project\\Figures\\'
        'step1_SYMH_prediction_2001_storm3.png')

plt.legend()
plt.savefig(path)



#%% Storm4 Forecast
    
# Initially, let's act under the assumption of no time delay from bow shock to Earth

# Filter the DataFrame
df_storm4 = df_params[(275 < df_params['Day']) & (df_params['Day'] < 279)]

sym_forecast_storm4 = []
current_sym = df_storm4['SYM/H, nT'].tolist()[0]

for i in range(len(df_storm4['Day'].tolist())-1):
    
    new_sym = SYM_forecast(current_sym, dt, a, b, c, d,
                                     df_storm4['Flow pressure, nPa'].tolist()[i],
                                     df_storm4['Flow pressure, nPa'].tolist()[i+1],
                                     df_storm4['Electric field, mV/m'].tolist()[i])
    sym_forecast_storm4.append(new_sym)
    current_sym = new_sym
    


#%% Storm4 Plot 

# Extract relevant columns
days4 = df_storm4['DayHourMin'].values
sym_storm4 = df_storm4['SYM/H, nT'].values

sym_forecast_storm4[0] = sym_storm4[0]   # Add initial valu we use to propagate through forecast so
                                         # arrays are same length, ready for cross-correlation analysis

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(days4, sym_storm4, label='SYM/H')
plt.grid()

# Adding labels and title
plt.xlabel('Day in 2001', fontsize=15)
plt.ylabel('SYM/H (nT)', fontsize=15)
#plt.title('Storm 4', fontsize=15)

# Set x-axis ticks to display only whole numbers
plt.xticks(range(int(days4[0]), int(days4[-1]) + 1), fontsize=15)
plt.yticks(fontsize=15)

# Plot forecasted SYM/H in storm to compare to calculated SYM/H
plt.plot(days4[1:], sym_forecast_storm4, label='Forecasted SYM/H')

path = ('C:\\Users\\logan\\OneDrive - Imperial College London\\Uni\\Year 4\\MSci Project\\Figures\\'
        'step1_SYMH_prediction_2001_storm4.png')

plt.legend(fontsize=15)
plt.savefig(path)



#%% Cross-correlations to find a time shift from bow shock to Earth

# Storm1

from scipy.signal import correlate

# Perform cross-correlation
lags = np.arange(-len(sym_storm1) + 1, len(sym_storm1))
cross_corr_values = correlate(sym_storm1, sym_forecast_storm1, mode='full') / len(sym_storm1)

# Calculate corresponding time delays
time_delays = lags * dt

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
lower_lim = -500*60   # x60 because graph shows in minutes but data is in seconds, so convert mins-->secs
upper_lim = 700*60

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


#%% Storm2

# Perform cross-correlation
lags = np.arange(-len(sym_storm2) + 1, len(sym_storm2))
cross_corr_values = correlate(sym_storm2, sym_forecast_storm2, mode='full') / len(sym_storm2)

# Calculate corresponding time delays
time_delays = lags * dt

# Find the index of the maximum cross-correlation value
peak_index = np.argmax(np.abs(cross_corr_values))

# Plot the cross-correlation values
plt.figure(figsize=(12, 6))
plt.plot(time_delays[:-1]/60, cross_corr_values, label='Cross-Correlation')
plt.axvline(time_delays[peak_index]/60, color='red', linestyle='--', label='Peak Correlation')
plt.xlabel('Time Delay (minutes)', fontsize=15)
plt.ylabel('Cross-Correlation', fontsize=15)
plt.title('Storm2')
plt.grid(True)
#plt.text(-1000,7200,'Peak Cross-Correlation is Observed at \n 65 Minutes After Bow Shock Measurements',
#         horizontalalignment='right')


# Fit a Gaussian to the top (or all?) of the curve to try and quantify a time shift & its uncertainty

popt, pcov = curve_fit(Gaussian, time_delays[:-1]/60, cross_corr_values)

plt.plot(time_delays[:-1]/60, Gaussian(time_delays[:-1]/60, *popt), label='Gaussian Fit')

plt.legend()
plt.show()

print('\nPeak cross-correlation is observed', int(time_delays[peak_index]/60), 'minutes after bow shock measurement')
print('\nGaussian fit parameters:')
print('A =',popt[0],'\nmu =',popt[1],'\nsigma =',popt[2])


#%% Storm3

# Perform cross-correlation
lags = np.arange(-len(sym_storm3) + 1, len(sym_storm3))
cross_corr_values = correlate(sym_storm3, sym_forecast_storm3, mode='full') / len(sym_storm3)

# Calculate corresponding time delays
time_delays = lags * dt

# Find the index of the maximum cross-correlation value
peak_index = np.argmax(np.abs(cross_corr_values))

# Plot the cross-correlation values
plt.figure(figsize=(12, 6))
plt.plot(time_delays[:-1]/60, cross_corr_values, label='Cross-Correlation')
plt.axvline(time_delays[peak_index]/60, color='red', linestyle='--', label='Peak Correlation')
plt.xlabel('Time Delay (minutes)', fontsize=15)
plt.ylabel('Cross-Correlation', fontsize=15)
plt.title('Storm3')
plt.grid(True)
#plt.text(-1000,7200,'Peak Cross-Correlation is Observed at \n 65 Minutes After Bow Shock Measurements',
#         horizontalalignment='right')


# Fit a Gaussian to the top (or all?) of the curve to try and quantify a time shift & its uncertainty

popt, pcov = curve_fit(Gaussian, time_delays[:-1]/60, cross_corr_values)

plt.plot(time_delays[:-1]/60, Gaussian(time_delays[:-1]/60, *popt), label='Gaussian Fit')

plt.legend()
plt.show()

print('\nPeak cross-correlation is observed', int(time_delays[peak_index]/60), 'minutes after bow shock measurement')
print('\nGaussian fit parameters:')
print('A =',popt[0],'\nmu =',popt[1],'\nsigma =',popt[2])


#%% Storm4

# Perform cross-correlation
lags = np.arange(-len(sym_storm4) + 1, len(sym_storm4))
cross_corr_values = correlate(sym_storm4, sym_forecast_storm4, mode='full') / len(sym_storm4)

# Calculate corresponding time delays
time_delays = lags * dt

# Find the index of the maximum cross-correlation value
peak_index = np.argmax(np.abs(cross_corr_values))

# Plot the cross-correlation values
plt.figure(figsize=(12, 6))
plt.plot(time_delays[:-1]/60, cross_corr_values, label='Cross-Correlation')
plt.axvline(time_delays[peak_index]/60, color='red', linestyle='--', label='Peak Correlation')
plt.xlabel('Time Delay (minutes)', fontsize=15)
plt.ylabel('Cross-Correlation', fontsize=15)
plt.title('Storm4')
plt.grid(True)
#plt.text(-1000,7200,'Peak Cross-Correlation is Observed at \n 65 Minutes After Bow Shock Measurements',
#         horizontalalignment='right')


# Fit a Gaussian to the top (or all?) of the curve to try and quantify a time shift & its uncertainty

popt, pcov = curve_fit(Gaussian, time_delays[:-1]/60, cross_corr_values)

plt.plot(time_delays[:-1]/60, Gaussian(time_delays[:-1]/60, *popt), label='Gaussian Fit')

plt.legend()
plt.show()

print('\nPeak cross-correlation is observed', int(time_delays[peak_index]/60), 'minutes after bow shock measurement')
print('\nGaussian fit parameters:')
print('A =',popt[0],'\nmu =',popt[1],'\nsigma =',popt[2])


#%% Test model over whole year and produce some criteria to quantify an event

# This ones very slow but kept in case of error with vectorised method in next section

from tqdm import tqdm

sym_forecast_2001 = []
current_sym = df_params['SYM/H, nT'].tolist()[0]

# Use tqdm to create a progress bar
for i in tqdm(range(len(df_params['Day'].tolist()) - 1), desc="Forecasting SYM/H 2001"):
    new_sym = SYM_forecast(current_sym, dt, a, b, c, d,
                            df_params['Flow pressure, nPa'].tolist()[i],
                            df_params['Flow pressure, nPa'].tolist()[i + 1],
                            df_params['Electric field, mV/m'].tolist()[i])
    sym_forecast_2001.append(new_sym)
    current_sym = new_sym

#%% Optimisation test as above will take about 7 hours - WOW this took seconds

###############################
# AHHHHHHHHHHHHHHHHHHHHHHH WHY DOESNT THIS WORK NOW, LIST ERROR ?!?!?!?!? - think you have to run all again if
# this happens
###############################

from tqdm import tqdm

# Extract relevant columns as NumPy arrays
sym_2001 = df_params['SYM/H, nT'].values

pressure = df_params['Flow pressure, nPa'].values
Efield = df_params['Electric field, mV/m'].values

# Initialise empty forecast array and add initial value
sym_forecast_2001 = np.empty_like(sym_2001)
sym_forecast_2001[0] = sym_2001[0]
sym_forecast_2001 = np.asarray(sym_forecast_2001)

# Vectorized computation
for i in tqdm(range(len(sym_2001) - 1), desc="Forecasting SYM/H 2001"):
    sym_forecast_2001[i + 1] = SYM_forecast(sym_forecast_2001[i], dt, a, b, c, d,
                                            pressure[i], pressure[i + 1],
                                            Efield[i])

# Note: The first element of sym_forecast_2001 remains the same as the original data


#%% 2001 Plot 

# Extract relevant columns
days2001 = df_params['DayHourMin'].values
sym_2001 = df_params['SYM/H, nT'].values

# New vectorised function does this in above section:
#sym_forecast_2001[0] = sym_2001[0]   # Add initial valu we use to propagate through forecast so
                                      # arrays are same length, ready for cross-correlation analysis

# Plotting
plt.figure(figsize=(12, 6))
#plt.plot(days2001, sym_2001, label = 'SYM/H')
plt.grid()

# Adding labels and title
plt.xlabel('Day in 2001', fontsize=15)
plt.ylabel('SYM/H (nT)', fontsize=15)
#plt.title('Storm 3', fontsize=15)

# Set x-axis ticks to display only whole numbers
#plt.xticks(range(int(days2001[0]), int(days2001[-1]) + 1), fontsize=15)
plt.yticks(fontsize=15)

# Plot forecasted SYM/H in storm to compare to calculated SYM/H
#plt.plot(days2001, sym_forecast_2001, label = 'Forecasted SYM/H')

# Plot between desired x values
x_start = 101
x_end = 104

plt.plot(days2001[(days2001 >= x_start) & (days2001 <= x_end)], 
         sym_2001[(days2001 >= x_start) & (days2001 <= x_end)], 
         label='Forecasted SYM/H')

plt.plot(days2001[(days2001 >= x_start) & (days2001 <= x_end)], 
         sym_forecast_2001[(days2001 >= x_start) & (days2001 <= x_end)], 
         label='Forecasted SYM/H')

plt.legend()


#%% Cross-correlations


# Perform cross-correlation
lags = np.arange(-len(sym_2001) + 1, len(sym_2001))
cross_corr_values = correlate(sym_2001, sym_forecast_2001, mode='full') / len(sym_2001)

# Calculate corresponding time delays
time_delays = lags * dt

# Find the index of the maximum cross-correlation value
peak_index = np.argmax(np.abs(cross_corr_values))

# Plot the cross-correlation values
plt.figure(figsize=(12, 6))
plt.plot(time_delays/60, cross_corr_values, label='Cross-Correlation')
#plt.axvline(time_delays[peak_index]/60, color='red', linestyle='--', label='Peak Correlation')
plt.xlabel('Time Delay (minutes)', fontsize=15)
plt.ylabel('Cross-Correlation', fontsize=15)
#plt.title('Cross-Correlation between SYM/H and Forecasted SYM/H', fontsize=15)
plt.legend()
plt.grid(True)


#%% Now come up with criteria for a storm and compare with measured number of storms from 2001

# Storm classification is this (nT): -50  > s >-100 = moderate
#                                    -100 > s >-250 = intense
#                                    -250 > s       = superstorm     


# Define the storm classification criteria
moderate_condition = (-50 <= sym_forecast_2001) & (sym_forecast_2001 >= -100)
intense_condition = (-100 > sym_forecast_2001) & (sym_forecast_2001 > -250)
superstorm_condition = (sym_forecast_2001 <= -250)

# Create an array to store storm classifications
storm_classification = np.zeros_like(sym_forecast_2001, dtype=int)

# Apply the storm classifications
storm_classification[moderate_condition] = 1  # Moderate
storm_classification[intense_condition] = 2   # Intense
storm_classification[superstorm_condition] = 3  # Superstorm

# Count the number of storms in each category
num_moderate_storms = np.count_nonzero(storm_classification == 1)
num_intense_storms = np.count_nonzero(storm_classification == 2)
num_superstorms = np.count_nonzero(storm_classification == 3)

print(f"Number of Moderate Storms: {num_moderate_storms}")
print(f"Number of Intense Storms: {num_intense_storms}")
print(f"Number of Superstorms: {num_superstorms}")


