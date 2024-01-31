# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:07:57 2024

@author: logan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from forecasting_model import SYM_forecast


##################################################### SPACECRAFT 1 #################################################
####################################################################################################################

# Read in B field and plasma data for the FIRST dataset
df_params = pd.read_csv('dscovr_2019_1min.csv')

#%% Clean Data

# Identify rows with NaN values from USEFUL columns
df_params['BZ, GSE, nT'] = df_params['BZ, GSE, nT'].replace(9999.990, np.nan)
df_params['Speed, km/s'] = df_params['Speed, km/s'].replace(99999.900, np.nan)
df_params['Proton Density, n/cc'] = df_params['Proton Density, n/cc'].replace(999.999, np.nan)

# Combine 'Day' and 'Hour' into a new column 'DayHour'
df_params['DayHourMin'] = df_params['Day'] + df_params['Hour'] / 24.0 + df_params['Minute'] / 60 / 24
days = df_params['DayHourMin'].values

# Check where values will be interpolated
plt.plot(days,df_params['BZ, GSE, nT'])

#%%

# Interpolate NaN values
for column in df_params.columns[3:]:  # Exclude the time columns
    df_params[column] = df_params[column].interpolate()

# Drop rows with remaining NaN values
df_params = df_params.dropna()

# Isolate the different USEFUL columns
Bz = df_params['BZ, GSE, nT'].values
vtot = df_params['Speed, km/s'].values
pdens = df_params['Proton Density, n/cc'].values

# Combine 'Day' and 'Hour' into a new column 'DayHour'
df_params['DayHourMin'] = df_params['Day'] + df_params['Hour'] / 24.0 + df_params['Minute'] / 60 / 24
days = df_params['DayHourMin'].values

#%% Plot Bz as a test over 2019 storm range

# Choose days of storm
lower_day = 132
upper_day = 140

# Filter the DataFrame
df_storm19 = df_params[(lower_day < df_params['Day']) & (df_params['Day'] < upper_day)]

days_storm = df_storm19['DayHourMin'].values
Bz_storm = df_storm19['BZ, GSE, nT'].values

plt.plot(days_storm,Bz_storm)


#%% Extract quantities needed for SYM/H prediction and assign functions

def P(proton_density,velocity_mag):
    # Found the correct unit scaling on OMNIweb using units data is given in!
    return proton_density*velocity_mag**2*2e-6

def E(velocity_mag,Bz):
    
    return velocity_mag*Bz*1e-3

vtot_storm = df_storm19['Speed, km/s'].values
pdens_storm = df_storm19['Proton Density, n/cc'].values
Bz_storm = df_storm19['BZ, GSE, nT'].values

P_storm = P(pdens_storm,vtot_storm)
E_storm = E(vtot_storm,Bz_storm)

# Plot E and P to observe distributions

plt.plot(days_storm,P_storm,label='P')
plt.plot(days_storm,E_storm,label='E')
plt.xlabel('Day in 2019')
plt.ylabel('E (mV/m) & P (nPa)')
plt.grid()

plt.legend()


#%% Use model to forecast SYM/H, starting with a basic model without propagation

# Import initial SYM/H value

df_sym = pd.read_csv('omni_ind_params_2019_1min.csv')

# Filter the DataFrame
df_sym_storm = df_sym[(lower_day < df_sym['Day']) & (df_sym['Day'] < upper_day)]
sym_storm = df_sym_storm['SYM/H, nT'].values

sym_forecast_storm = []
initial_sym = sym_storm[0]  # Record this for later, as current_sym will change
current_sym = sym_storm[0]

for i in range(len(df_storm19['Day'].tolist())-1):
    
    new_sym = SYM_forecast(current_sym,
                                     P_storm[i],
                                     P_storm[i+1],
                                     E_storm[i])
    sym_forecast_storm.append(new_sym)
    current_sym = new_sym
    

sym_forecast_storm.insert(0,initial_sym)   # Add initial value we used to propagate through forecast


#%% Plot results

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(days_storm, sym_storm, label = 'Measured SYM/H')
plt.grid()

# Adding labels and title
plt.xlabel('Day in 2019', fontsize=15)
plt.ylabel('SYM/H (nT)', fontsize=15)
#plt.title('Storm 1', fontsize=15)

# Set x-axis ticks to display only whole numbers
plt.xticks(range(int(days_storm[0]), int(days_storm[-1]) + 1), fontsize=15)
plt.yticks(fontsize=15)

# Plot forecasted SYM/H in storm to compare to calculated SYM/H
plt.plot(days_storm, sym_forecast_storm, label = 'DSCOVR Forecasted SYM/H')

path = ('C:\\Users\\logan\\OneDrive - Imperial College London\\Uni\\Year 4\\MSci Project\\Figures\\'
        'step1_SYMH_prediction_2001_storm1.png')

plt.legend(loc='lower right',fontsize=15)
#plt.savefig(path)
plt.show()




##################################################### SPACECRAFT 2 #################################################
#%% ################################################################################################################


# Read in B field and plasma data for the NEW dataset
df_params2 = pd.read_csv('wind_L1_2019_1min.csv')  # Change the filename to the new dataset

#%% Clean Data for the new dataset

# Identify rows with NaN values
df_params2['BZ, GSE, nT'] = df_params2['BZ, GSE, nT'].replace(9999.99, np.nan)
df_params2['Kp_proton Density, n/cc'] = df_params2['Kp_proton Density, n/cc'].replace(999.99, np.nan)
df_params2['KP_Speed, km/s'] = df_params2['KP_Speed, km/s'].replace(99999.9, np.nan)

# Combine 'Day' and 'Hour' into a new column 'DayHour'
df_params2['DayHourMin'] = df_params2['Day'] + df_params2['Hour'] / 24.0 + df_params2['Minute'] / 60 / 24
days2 = df_params2['DayHourMin'].values

# Check where values will be interpolated
plt.plot(days2, df_params2['BZ, GSE, nT'])

#%%

# Interpolate NaN values for the new dataset
for column in df_params2.columns[3:]:  # Exclude the time columns
    df_params2[column] = df_params2[column].interpolate()

# Drop rows with remaining NaN values
df_params2 = df_params2.dropna()

# Isolate the different USEFUL columns for the new dataset
Bz2 = df_params2['BZ, GSE, nT'].values
vtot2 = df_params2['KP_Speed, km/s'].values  # Change here
pdens2 = df_params2['Kp_proton Density, n/cc'].values  # Change here

# Combine 'Day' and 'Hour' into a new column 'DayHour'
df_params2['DayHourMin'] = df_params2['Day'] + df_params2['Hour'] / 24.0 + df_params2['Minute'] / 60 / 24
days2 = df_params2['DayHourMin'].values

#%% Plot Bz for the new dataset as a test

plt.plot(days2, df_params2['BZ, GSE, nT'])
plt.xlabel('Day in 2019')
plt.ylabel('BZ, GSE, nT')
plt.title('BZ for the New Dataset')
plt.show()

#%% Compare the two datasets

# Choose days of storm for the new dataset
lower_day2 = 132
upper_day2 = 140

# Filter the DataFrame for the new dataset
df_storm19_2 = df_params2[(lower_day2 < df_params2['Day']) & (df_params2['Day'] < upper_day2)]

days_storm2 = df_storm19_2['DayHourMin'].values
Bz_storm2 = df_storm19_2['BZ, GSE, nT'].values

# Plot Bz for both datasets
plt.plot(days_storm, Bz_storm, label='BZ, GSE, nT - Original Dataset')
plt.plot(days_storm2, Bz_storm2, label='BZ, GSE, nT - New Dataset')
plt.xlabel('Day in 2019')
plt.ylabel('BZ, GSE, nT')
plt.title('Comparison of BZ for Original and New Datasets')
plt.legend()
plt.show()


#%% Extract quantities needed for SYM/H prediction and assign functions

def P2(proton_density, velocity_mag):
    return proton_density * velocity_mag**2 * 2e-6

def E2(velocity_mag, Bz):
    return -velocity_mag * Bz * 1e-3

# Extract data for the new dataset
vtot_storm2 = df_storm19_2['KP_Speed, km/s'].values
pdens_storm2 = df_storm19_2['Kp_proton Density, n/cc'].values
Bz_storm2 = df_storm19_2['BZ, GSE, nT'].values

P_storm2 = P2(pdens_storm2, vtot_storm2)
E_storm2 = E2(vtot_storm2, Bz_storm2)

# Plot E and P to observe distributions for the new dataset
plt.plot(days_storm2, P_storm2, label='P - New Dataset 2')
plt.plot(days_storm2, E_storm2, label='E - New Dataset 2')
plt.xlabel('Day in 2019')
plt.ylabel('E (mV/m) & P (nPa)')
plt.grid()
plt.legend()
plt.show()

#%% Use model to forecast SYM/H for the new dataset

# Import initial SYM/H value
df_sym2 = pd.read_csv('omni_ind_params_2019_1min.csv')

# Filter the DataFrame for the new dataset
df_sym_storm2 = df_sym2[(lower_day2 < df_sym2['Day']) & (df_sym2['Day'] < upper_day2)]
sym_storm2 = df_sym_storm2['SYM/H, nT'].values

sym_forecast_storm2 = []
initial_sym2 = sym_storm2[0]
current_sym2 = sym_storm2[0]

for i in range(len(df_storm19_2['Day'].tolist()) - 1):
    new_sym2 = SYM_forecast(current_sym2, P_storm2[i], P_storm2[i + 1], E_storm2[i])
    sym_forecast_storm2.append(new_sym2)
    current_sym2 = new_sym2

sym_forecast_storm2.insert(0, initial_sym2)

# Plot results for the new dataset
plt.figure(figsize=(12, 6))
plt.plot(days_storm2, sym_storm2, label='Measured SYM/H')
plt.grid()
plt.xlabel('Day in 2019', fontsize=15)
plt.ylabel('SYM/H (nT)', fontsize=15)
plt.xticks(range(int(days_storm2[0]), int(days_storm2[-1]) + 1), fontsize=15)
plt.yticks(fontsize=15)
plt.plot(days_storm2, sym_forecast_storm2, label='Wind Forecasted SYM/H')

plt.legend(loc='lower right', fontsize=15)
plt.show()






##################################################### SPACECRAFT 3 #################################################
#%% ################################################################################################################

# Read in B field and plasma data for the NEW dataset
df_params3 = pd.read_csv('ace_L1_2019_1min.csv')  # Change the filename to the new dataset

#%% Clean Data for the new dataset

# Identify rows with NaN values
df_params3['BZ, GSE, nT'] = df_params3['BZ, GSE, nT'].replace(9999.99, np.nan)
df_params3['Kp_proton Density, n/cc'] = df_params3['Kp_proton Density, n/cc'].replace(999.99, np.nan)
df_params3['KP_Speed, km/s'] = df_params3['KP_Speed, km/s'].replace(99999.9, np.nan)  # Change here

# Combine 'Day' and 'Hour' into a new column 'DayHour'
df_params3['DayHourMin'] = df_params3['Day'] + df_params3['Hour'] / 24.0 + df_params3['Minute'] / 60 / 24
days3 = df_params3['DayHourMin'].values

# Check where values will be interpolated
plt.plot(days3, df_params3['BZ, GSE, nT'])

#%%

# Interpolate NaN values for the new dataset
for column in df_params3.columns[3:]:  # Exclude the time columns
    df_params3[column] = df_params3[column].interpolate()

# Drop rows with remaining NaN values
df_params3 = df_params3.dropna()

# Isolate the different USEFUL columns for the new dataset
Bz3 = df_params3['BZ, GSE, nT'].values
vtot3 = df_params3['KP_Speed, km/s'].values  # Change here
pdens3 = df_params3['KP_proton Density, n/cc'].values  # Change here

# Combine 'Day' and 'Hour' into a new column 'DayHour'
df_params3['DayHourMin'] = df_params3['Day'] + df_params3['Hour'] / 24.0 + df_params3['Minute'] / 60 / 24
days3 = df_params3['DayHourMin'].values

#%% Plot Bz for the new dataset as a test

plt.plot(days3, df_params3['BZ, GSE, nT'])
plt.xlabel('Day in 2019')
plt.ylabel('BZ, GSE, nT')
plt.title('BZ for the New Dataset')
plt.show()

#%% Compare the two datasets

# Choose days of storm for the new dataset
lower_day3 = 132
upper_day3 = 140

# Filter the DataFrame for the new dataset
df_storm19_3 = df_params3[(lower_day3 < df_params3['Day']) & (df_params3['Day'] < upper_day3)]

days_storm3 = df_storm19_3['DayHourMin'].values
Bz_storm3 = df_storm19_3['BZ, GSE, nT'].values

# Plot Bz for all three datasets
plt.plot(days_storm, Bz_storm, label='BZ, GSE, nT - Original Dataset')
plt.plot(days_storm2, Bz_storm2, label='BZ, GSE, nT - New Dataset 2')
plt.plot(days_storm3, Bz_storm3, label='BZ, GSE, nT - New Dataset 3')
plt.xlabel('Day in 2019')
plt.ylabel('BZ, GSE, nT')
plt.title('Comparison of BZ for All Datasets')
plt.legend()
plt.show()




#%% Extract quantities needed for SYM/H prediction and assign functions

def P3(proton_density, velocity_mag):
    return proton_density * velocity_mag**2 * 2e-6

def E3(velocity_mag, Bz):
    return -velocity_mag * Bz * 1e-3

# Extract data for the new dataset
vtot_storm3 = df_storm19_3['KP_Speed, km/s'].values
pdens_storm3 = df_storm19_3['Kp_proton Density, n/cc'].values
Bz_storm3 = df_storm19_3['BZ, GSE, nT'].values

P_storm3 = P3(pdens_storm3, vtot_storm3)
E_storm3 = E3(vtot_storm3, Bz_storm3)

# Plot E and P to observe distributions for the new dataset
plt.plot(days_storm3, P_storm3, label='P - New Dataset 3')
plt.plot(days_storm3, E_storm3, label='E - New Dataset 3')
plt.xlabel('Day in 2019')
plt.ylabel('E (mV/m) & P (nPa)')
plt.grid()
plt.legend()
plt.show()

#%% Use model to forecast SYM/H for the new dataset

# Import initial SYM/H value
df_sym3 = pd.read_csv('omni_ind_params_2019_1min.csv')

# Filter the DataFrame for the new dataset
df_sym_storm3 = df_sym3[(lower_day3 < df_sym3['Day']) & (df_sym3['Day'] < upper_day3)]
sym_storm3 = df_sym_storm3['SYM/H, nT'].values

sym_forecast_storm3 = []
initial_sym3 = sym_storm3[0]
current_sym3 = sym_storm3[0]

for i in range(len(df_storm19_3['Day'].tolist()) - 1):
    new_sym3 = SYM_forecast(current_sym3, P_storm3[i], P_storm3[i + 1], E_storm3[i])
    sym_forecast_storm3.append(new_sym3)
    current_sym3 = new_sym3

sym_forecast_storm3.insert(0, initial_sym3)

# Plot results for the new dataset
plt.figure(figsize=(12, 6))
plt.plot(days_storm3, sym_storm3, label='Measured SYM/H')
plt.grid()
plt.xlabel('Day in 2019', fontsize=15)
plt.ylabel('SYM/H (nT)', fontsize=15)
plt.xticks(range(int(days_storm3[0]), int(days_storm3[-1]) + 1), fontsize=15)
plt.yticks(fontsize=15)
plt.plot(days_storm3, sym_forecast_storm3, label='Wind Forecasted SYM/H')

plt.legend(loc='lower right', fontsize=15)
plt.show()
