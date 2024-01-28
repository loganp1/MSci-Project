# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:07:57 2024

@author: logan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from forecasting_model import SYM_forecast

# Read in B field and plasma data
df_params = pd.read_csv('dscovr_2019_1min.csv')

#%% Clean Data

# Identify rows with NaN values
df_params['BX, GSE, nT'] = df_params['BX, GSE, nT'].replace(9999.990, np.nan)
df_params['BY, GSE, nT'] = df_params['BY, GSE, nT'].replace(9999.990, np.nan)
df_params['BZ, GSE, nT'] = df_params['BZ, GSE, nT'].replace(9999.990, np.nan)
df_params['Vector Mag.,nT'] = df_params['Vector Mag.,nT'].replace(9999.990, np.nan)
df_params['Vx Velocity,km/s'] = df_params['Vx Velocity,km/s'].replace(99999.900, np.nan)
df_params['Vy Velocity,km/s'] = df_params['Vx Velocity,km/s'].replace(99999.900, np.nan)
df_params['Vz Velocity,km/s'] = df_params['Vx Velocity,km/s'].replace(99999.900, np.nan)
df_params['Speed, km/s'] = df_params['Speed, km/s'].replace(99999.900, np.nan)
df_params['Proton Density, n/cc'] = df_params['Proton Density, n/cc'].replace(999.999, np.nan)
df_params['Wind, Xgse,Re'] = df_params['Wind, Xgse,Re'].replace(99999.990, np.nan)
df_params['Wind, Ygse,Re'] = df_params['Wind, Ygse,Re'].replace(99999.990, np.nan)
df_params['Wind, Zgse,Re'] = df_params['Wind, Zgse,Re'].replace(99999.990, np.nan)

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

# Isolate the different columns
Bx = df_params['BX, GSE, nT'].values
By = df_params['BY, GSE, nT'].values
Bz = df_params['BZ, GSE, nT'].values
Btot = df_params['Vector Mag.,nT'].values
vx = df_params['Vx Velocity,km/s'].values
vy = df_params['Vy Velocity,km/s'].values
vz = df_params['Vz Velocity,km/s'].values
vtot = df_params['Speed, km/s'].values
pdens = df_params['Proton Density, n/cc'].values
xgse = df_params['Wind, Xgse,Re'].values
ygse = df_params['Wind, Ygse,Re'].values
zgse = df_params['Wind, Zgse,Re'].values

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
    
    #m = 1.6726e-27
    
    #return proton_density*m*velocity_mag**2*1e21  # 10^21 to get into nPa units as needed for SYM/H
    #return proton_density*velocity_mag*10e-3      # Comes from 10^9 for nano & 10^12 from derived from units
    
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
#plt.plot(days_storm, sym_forecast_storm, label = 'Forecasted SYM/H')

#path = ('C:\\Users\\logan\\OneDrive - Imperial College London\\Uni\\Year 4\\MSci Project\\Figures\\'
#        'step1_SYMH_prediction_2001_storm1.png')

#plt.plot(days_storm,P_storm/100,label='P')
#plt.plot(days_storm,E_storm/100,label='E')

plt.legend(loc='lower right',fontsize=15)
#plt.savefig(path)
plt.show()


#%% Since the test above gives strange results after around 134.8 days, descending into another 'storm' where
### none exists and bow shock model doesn't predict, I will update the starting value of SYM/H to just before
### this period to see what happens

