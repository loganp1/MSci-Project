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


# Model from Burton et al. 1975
def SYM_forecast(SYM_i, dt, a, b, c, d, P_i, P_iplus1, E_i):
    
    derivative_term = b * (P_iplus1**0.5 - P_i**0.5)/dt
    deriv_term.append(derivative_term*dt)
    F_E.append(F(E_i,d)*dt)
    term1.append(-a*(SYM_i - b * np.sqrt(P_i) + c)*dt)
    
    return SYM_i + (-a * (SYM_i - b * np.sqrt(P_i) + c) + F(E_i, d) + derivative_term) * dt


#%% Plot E and P distributions for visual comparison vs SYM/H

# Create plot of storm1 in 2001 SYM/H data

# Combine 'Day' and 'Hour' into a new column 'DayHour'
df_params['DayHourMin'] = df_params['Day'] + df_params['Hour'] / 24.0 + df_params['Minute'] / 60 / 24

# Filter the DataFrame
df_storm1 = df_params[(132 < df_params['Day']) & (df_params['Day'] < 135)]

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
d = -1e-3 * gamma

#%% Storm1 Forecast
    
# Initially, let's act under the assumption of no time delay from bow shock to Earth

# Filter the DataFrame
df_storm1 = df_params[(132 < df_params['Day']) & (df_params['Day'] < 136)]

sym_forecast_storm1 = []
current_sym = df_storm1['SYM/H, nT'].tolist()[0]

# Reset the arrays for the different terms in the model
deriv_term = []
F_E = []
term1 = []

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


