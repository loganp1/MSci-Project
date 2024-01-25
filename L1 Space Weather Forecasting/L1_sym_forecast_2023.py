# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:21:39 2024

@author: logan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from forecasting_model import SYM_forecast

df_wind23 = pd.read_csv('wind_2023_1min.csv')

# Clean Data

# Identify rows with NaN values
df_wind23['BX, GSE, nT'] = df_wind23['BX, GSE, nT'].replace(9999.990000, np.nan)
df_wind23['BY, GSE, nT'] = df_wind23['BY, GSE, nT'].replace(9999.990000, np.nan)
df_wind23['BZ, GSE, nT'] = df_wind23['BZ, GSE, nT'].replace(9999.990000, np.nan)
df_wind23['Vector Mag.,nT'] = df_wind23['Vector Mag.,nT'].replace(9999.990000, np.nan)
df_wind23['Field Magnitude,nT'] = df_wind23['Field Magnitude,nT'].replace(9999.990000, np.nan)
df_wind23['KP_Vx,km/s'] = df_wind23['KP_Vx,km/s'].replace(99999.900000, np.nan)
df_wind23['Kp_Vy, km/s'] = df_wind23['Kp_Vy, km/s'].replace(99999.900000, np.nan)
df_wind23['KP_Vz, km/s'] = df_wind23['KP_Vz, km/s'].replace(99999.900000, np.nan)
df_wind23['KP_Speed, km/s'] = df_wind23['KP_Speed, km/s'].replace(99999.900000, np.nan)
df_wind23['Kp_proton Density, n/cc'] = df_wind23['Kp_proton Density, n/cc'].replace(999.990000, np.nan)
df_wind23['Wind, Xgse,Re'] = df_wind23['Wind, Xgse,Re'].replace(9999.990000, np.nan)
df_wind23['Wind, Ygse,Re'] = df_wind23['Wind, Ygse,Re'].replace(9999.990000, np.nan)
df_wind23['Wind, Zgse,Re'] = df_wind23['Wind, Zgse,Re'].replace(9999.990000, np.nan)

# Interpolate NaN values
for column in df_wind23.columns[3:]:  # Exclude the time columns
    df_wind23[column] = df_wind23[column].interpolate()

# Drop rows with remaining NaN values
df_wind23 = df_wind23.dropna()

# Isolate Bz
Bz = df_wind23['BZ, GSE, nT']

# Combine 'Day' and 'Hour' into a new column 'DayHour'
df_wind23['DayHourMin'] = df_wind23['Day'] + df_wind23['Hour'] / 24.0 + df_wind23['Minute'] / 60 / 24
days = df_wind23['DayHourMin'].values

plt.plot(days,Bz)

#%% Plot Bz as a test over 2019 storm range

# Choose days of storm
lower_day = 110
upper_day = 120

# Filter the DataFrame
df_wind23 = df_wind23[(lower_day < df_wind23['Day']) & (df_wind23['Day'] < upper_day)]

days_storm = df_wind23['DayHourMin'].values
Bz_storm = df_wind23['BZ, GSE, nT'].values

plt.plot(days_storm,Bz_storm)


#%% Extract quantities needed for SYM/H prediction and assign functions

def P(proton_density,velocity_mag):
    # Found the correct unit scaling on OMNIweb using units data is given in!
    return proton_density*velocity_mag**2*2e-6

def E(velocity_mag,Bz):
    
    return velocity_mag*Bz*1e-3

# Isolate the useful columns
Bz_storm = df_wind23['BZ, GSE, nT'].values
vtot_storm = df_wind23['KP_Speed, km/s'].values
pdens_storm = df_wind23['Kp_proton Density, n/cc'].values

P_storm = P(pdens_storm,vtot_storm)
E_storm = E(vtot_storm,Bz_storm)

# Plot E and P to observe distributions

plt.plot(days_storm,P_storm,label='P')
plt.plot(days_storm,E_storm,label='E')
#plt.plot(days_storm,vtot_storm)

plt.legend()


#%% Use model to forecast SYM/H, starting with a basic model without propagation

# Import initial SYM/H value

df_sym = pd.read_csv('omni_ind_params_BSN_2023_1min.csv')

# Filter the DataFrame
df_sym_storm = df_sym[(lower_day < df_sym['Day']) & (df_sym['Day'] < upper_day)]
sym_storm = df_sym_storm['SYM/H, nT'].values

sym_forecast_storm = []
initial_sym = sym_storm[0]  # Record this for later, as current_sym will change
current_sym = sym_storm[0]

for i in range(len(df_wind23['Day'].tolist())-1):
    
    new_sym = SYM_forecast(current_sym,
                                     P_storm[i],
                                     P_storm[i+1],
                                     E_storm[i])
    sym_forecast_storm.append(new_sym)
    current_sym = new_sym
    

sym_forecast_storm.insert(0,initial_sym)   # Add initial value we used to propagate through forecast

# Divide by 1000 as haven't got units figured out yet, but this is correct
#sym_forecast_storm = np.asarray(sym_forecast_storm)/1000

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
plt.plot(days_storm, sym_forecast_storm, label = 'Forecasted SYM/H')

#path = ('C:\\Users\\logan\\OneDrive - Imperial College London\\Uni\\Year 4\\MSci Project\\Figures\\'
#        'step1_SYMH_prediction_2001_storm1.png')

#plt.plot(days_storm,P_storm/100,label='P')
#plt.plot(days_storm,E_storm/100,label='E')

plt.legend(loc='lower left',fontsize=15)
#plt.savefig(path)
plt.show()