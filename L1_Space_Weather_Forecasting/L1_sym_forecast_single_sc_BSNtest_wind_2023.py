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

# Need unix time data to apply propagation methods
df_wind23_totransform = pd.read_csv('wind_2023_1min_unix.csv')

# Also need BSN data to propagate data to correct location
df_BSN = pd.read_csv('omni_ind_params_BSN_2023_1min_unix.csv')

# Combine 2 dfs into one as need same number of points for propagate function
combined_df = pd.merge(df_wind23_totransform, df_BSN, on='Time', how='outer')

#%%

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

    
# Repeat above for unix dataframe

# Identify rows with NaN values in df_wind23_totransform
df_wind23_totransform['BX, GSE, nT'] = df_wind23_totransform['BX, GSE, nT'].replace(9999.990000, np.nan)
df_wind23_totransform['BY, GSE, nT'] = df_wind23_totransform['BY, GSE, nT'].replace(9999.990000, np.nan)
df_wind23_totransform['BZ, GSE, nT'] = df_wind23_totransform['BZ, GSE, nT'].replace(9999.990000, np.nan)
df_wind23_totransform['Vector Mag.,nT'] = df_wind23_totransform['Vector Mag.,nT'].replace(9999.990000, np.nan)
df_wind23_totransform['Field Magnitude,nT'] = df_wind23_totransform['Field Magnitude,nT'].replace(9999.990000, np.nan)
df_wind23_totransform['KP_Vx,km/s'] = df_wind23_totransform['KP_Vx,km/s'].replace(99999.900000, np.nan)
df_wind23_totransform['Kp_Vy, km/s'] = df_wind23_totransform['Kp_Vy, km/s'].replace(99999.900000, np.nan)
df_wind23_totransform['KP_Vz, km/s'] = df_wind23_totransform['KP_Vz, km/s'].replace(99999.900000, np.nan)
df_wind23_totransform['KP_Speed, km/s'] = df_wind23_totransform['KP_Speed, km/s'].replace(99999.900000, np.nan)
df_wind23_totransform['Kp_proton Density, n/cc'] = df_wind23_totransform['Kp_proton Density, n/cc'].replace(999.990000, np.nan)
df_wind23_totransform['Wind, Xgse,Re'] = df_wind23_totransform['Wind, Xgse,Re'].replace(9999.990000, np.nan)
df_wind23_totransform['Wind, Ygse,Re'] = df_wind23_totransform['Wind, Ygse,Re'].replace(9999.990000, np.nan)
df_wind23_totransform['Wind, Zgse,Re'] = df_wind23_totransform['Wind, Zgse,Re'].replace(9999.990000, np.nan)

# Interpolate NaN values
for column in df_wind23_totransform.columns[3:]:  # Exclude the time columns
    df_wind23_totransform[column] = df_wind23_totransform[column].interpolate()

# Drop rows with remaining NaN values
df_wind23_totransform = df_wind23_totransform.dropna()


# Now do the same for combined_df

# Identify rows with NaN values in combined_df - copy above
combined_df['BX, GSE, nT'] = combined_df['BX, GSE, nT'].replace(9999.990000, np.nan)
combined_df['BY, GSE, nT'] = combined_df['BY, GSE, nT'].replace(9999.990000, np.nan)
combined_df['BZ, GSE, nT'] = combined_df['BZ, GSE, nT'].replace(9999.990000, np.nan)
combined_df['Vector Mag.,nT'] = combined_df['Vector Mag.,nT'].replace(9999.990000, np.nan)
combined_df['Field Magnitude,nT'] = combined_df['Field Magnitude,nT'].replace(9999.990000, np.nan)
combined_df['KP_Vx,km/s'] = combined_df['KP_Vx,km/s'].replace(99999.900000, np.nan)
combined_df['Kp_Vy, km/s'] = combined_df['Kp_Vy, km/s'].replace(99999.900000, np.nan)
combined_df['KP_Vz, km/s'] = combined_df['KP_Vz, km/s'].replace(99999.900000, np.nan)
combined_df['KP_Speed, km/s'] = combined_df['KP_Speed, km/s'].replace(99999.900000, np.nan)
combined_df['Kp_proton Density, n/cc'] = combined_df['Kp_proton Density, n/cc'].replace(999.990000, np.nan)
combined_df['Wind, Xgse,Re'] = combined_df['Wind, Xgse,Re'].replace(9999.990000, np.nan)
combined_df['Wind, Ygse,Re'] = combined_df['Wind, Ygse,Re'].replace(9999.990000, np.nan)
combined_df['Wind, Zgse,Re'] = combined_df['Wind, Zgse,Re'].replace(9999.990000, np.nan)

# Now replace the BSN part of the dataframe
columns_to_replace = [
    'Field magnitude average, nT',
    'BX, nT (GSE, GSM)',
    'BY, nT (GSE)',
    'BZ, nT (GSE)',
    'Speed, km/s',
    'Vx Velocity,km/s',
    'Vy Velocity, km/s',
    'Vz Velocity, km/s',
    'Proton Density, n/cc',
    'Flow pressure, nPa',
    'Electric field, mV/m',
    'S/C, Xgse,Re',
    'S/C, Ygse,Re',
    'S/c, Zgse,Re',
    'BSN location, Xgse,Re',
    'BSN location, Ygse,Re',
    'BSN location, Zgse,Re',
    'SYM/H, nT'
]

replace_values = [9999.99, 9999.99, 9999.99, 9999.99, 99999.9, 99999.9, 99999.9, 99999.9, 
                  999.99, 99.99, 999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.99]

# Identify rows with NaN values
combined_df[columns_to_replace] = combined_df[columns_to_replace].replace(replace_values, np.nan)

# Interpolate NaN values
for column in combined_df.columns[1:]:  # Exclude the time column
    combined_df[column] =combined_df[column].interpolate()

# Drop rows with remaining NaN values
combined_df = combined_df.dropna()


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
    
    return -velocity_mag*Bz*1e-3

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


#%% Now, here I'm going to incorporate Ned's single spacecraft shift (I only need E and P shifted)

from single_spacecraft_shift import propagate

# Define the desired order of columns as required for Ned's function
desired_columns_order = [
    'Time',
    'Wind, Xgse,Re',
    'Wind, Ygse,Re',
    'Wind, Zgse,Re',
    'KP_Vx,km/s',
    'Kp_Vy, km/s',
    'KP_Vz, km/s',
    'BZ, GSE, nT',
    'KP_Speed, km/s',
    'Kp_proton Density, n/cc'
]

# Select only the desired columns and reorder them
#df_wind23_totransform = df_wind23_totransform[desired_columns_order]
df_L1part = combined_df[desired_columns_order]

# Vectorized operations to create new columns
#df_wind23_totransform['P'] = P(df_wind23_totransform['Kp_proton Density, n/cc'], 
#                               df_wind23_totransform['KP_Speed, km/s'])
#df_wind23_totransform['E'] = E(df_wind23_totransform['KP_Speed, km/s'], df_wind23_totransform['BZ, GSE, nT'])

df_L1part['P'] = P(combined_df['Kp_proton Density, n/cc'], 
                               combined_df['KP_Speed, km/s'])
df_L1part['E'] = E(combined_df['KP_Speed, km/s'], combined_df['BZ, GSE, nT'])

# Drop the original columns
#df_wind23_totransform.drop(['Kp_proton Density, n/cc', 'KP_Speed, km/s', 'BZ, GSE, nT'], axis=1, inplace=True)
combined_df.drop(['Kp_proton Density, n/cc', 'KP_Speed, km/s', 'BZ, GSE, nT'], axis=1, inplace=True)


# Transform the data to desired format
#array_wind23_totransform = df_wind23_totransform.to_numpy().T
array_L1part = df_L1part.to_numpy().T

#%%

# Need to import positions from BSN where we are propagating to
#df_BSN = pd.read_csv('omni_ind_params_BSN_2023_1min_unix.csv')

desired_columns = ['Time',
    'BSN location, Xgse,Re',
    'BSN location, Ygse,Re',
    'BSN location, Zgse,Re'
]

df_BSNpart = combined_df[desired_columns]

# Transform the data to desired format
array_BSN = df_BSNpart.to_numpy().T

# Excellent, now can apply to functions
#df_propagated = propagate(array_wind23_totransform,array_BSN)

#%%

df_propagated = propagate(array_L1part,array_BSN)



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