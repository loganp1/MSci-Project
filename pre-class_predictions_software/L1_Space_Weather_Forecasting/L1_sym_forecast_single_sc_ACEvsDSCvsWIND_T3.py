# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:01:35 2024

@author: logan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from forecasting_model import SYM_forecast
from scipy.optimize import curve_fit

#%%

# Import actual SYM/H data

df_sym = pd.read_csv('SYM3_unix.csv')

df_sym['DateTime'] = pd.to_datetime(df_sym['Time'], unit='s')

DateTime = df_sym['DateTime'].values
sym = df_sym['SYM/H, nT'].values

'''# Format the date labels and set their font size
date_format = DateFormatter("%b-%Y")
plt.gca().xaxis.set_major_formatter(date_format)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Adjust the number of ticks as needed
plt.gcf().autofmt_xdate()  # Rotate the date labels for better visibility
plt.gca().tick_params(axis='x', which='both', labelsize=15)  # Set font size for date labels'''

plt.plot(DateTime,sym)

#%%

##################################################### SPACECRAFT 1 #################################################
####################################################################################################################

# Read in B field and plasma data for the DSCOVR dataset
df_params1 = pd.read_csv('dscovr_T3_1min_unix.csv')

#%% Clean Data

# Identify rows with NaN values from USEFUL columns
df_params1['BZ, GSE, nT'] = df_params1['BZ, GSE, nT'].replace(9999.990, np.nan)
df_params1['Speed, km/s'] = df_params1['Speed, km/s'].replace(99999.900, np.nan)
df_params1['Proton Density, n/cc'] = df_params1['Proton Density, n/cc'].replace(999.999, np.nan)

df_params1['DateTime'] = pd.to_datetime(df_params1['Time'], unit='s')
DateTime1 = df_params1['DateTime']

# Check where values will be interpolated
Bz1 = df_params1['BZ, GSE, nT'].values
plt.plot(DateTime1,Bz1)

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
plt.plot(DateTime1,Bz1)


#%% Extract quantities needed for SYM/H prediction and assign functions

def P(proton_density,velocity_mag):
    # Found the correct unit scaling on OMNIweb using units data is given in!
    return proton_density*velocity_mag**2*2e-6

def E(velocity_mag,Bz):
    
    return -velocity_mag*Bz*1e-3

P1 = P(pdens1,vtot1)
E1 = E(vtot1,Bz1)

# Plot E and P to observe distributions

plt.plot(DateTime1,P1,label='P')
plt.plot(DateTime1,E1,label='E')
plt.xlabel('Date')
plt.ylabel('E (mV/m) & P (nPa)')
plt.grid()

plt.legend()


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
plt.xlabel('Date', fontsize=15)
plt.ylabel('SYM/H (nT)', fontsize=15)
plt.title('DSCOVR Prediction', fontsize=15)

# Set x-axis ticks to display only whole numbers
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Plot forecasted SYM/H in storm to compare to calculated SYM/H
plt.plot(DateTime1, sym_forecast1, label = 'DSCOVR Forecasted SYM/H')

plt.legend(fontsize=15)
plt.show()




##################################################### SPACECRAFT 2 #################################################
#%% ################################################################################################################


# Read in B field and plasma data for the WIND dataset
df_params2 = pd.read_csv('wind_T3_1min_unix.csv')

#%% Clean Data

# Identify rows with NaN values from USEFUL columns
df_params2['BZ, GSE, nT'] = df_params2['BZ, GSE, nT'].replace(9999.990, np.nan)
df_params2['KP_Speed, km/s'] = df_params2['KP_Speed, km/s'].replace(99999.900, np.nan)
df_params2['Kp_proton Density, n/cc'] = df_params2['Kp_proton Density, n/cc'].replace(999.99, np.nan)

df_params2['DateTime'] = pd.to_datetime(df_params2['Time'], unit='s')
DateTime2 = df_params2['DateTime']

# Check where values will be interpolated
Bz2 = df_params2['BZ, GSE, nT'].values
plt.plot(DateTime2, Bz2)

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
plt.plot(DateTime2, Bz2)
plt.show()
plt.plot(DateTime2, vtot2)
plt.show()
plt.plot(DateTime2, pdens2)
plt.show()

#%% Extract quantities needed for SYM/H prediction and apply functions

P2 = P(pdens2, vtot2)
E2 = E(vtot2, Bz2)

# Plot E and P to observe distributions

plt.plot(DateTime2, P2, label='P')
plt.plot(DateTime2, E2, label='E')
plt.xlabel('Date')
plt.ylabel('E (mV/m) & P (nPa)')
plt.grid()

plt.legend()

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
plt.xlabel('Date', fontsize=15)
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





##################################################### SPACECRAFT 3 #################################################
#%% ################################################################################################################

# Read in B field and plasma data for the WIND dataset
df_params3 = pd.read_csv('ace_T3_1min_unix.csv')

#%% Data already cleaned for ACE from SPEDAS

df_params3['DateTime'] = pd.to_datetime(df_params3['Time'], unit='s')
DateTime3 = df_params3['DateTime']

# Check where values will be interpolated
Bz3 = df_params3['Bz'].values
plt.plot(DateTime3, Bz3)

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
plt.plot(DateTime3, Bz3)
plt.show()
plt.plot(DateTime3, vtot3)
plt.show()
plt.plot(DateTime3, pdens3)
plt.show()

#%% Extract quantities needed for SYM/H prediction and apply functions

P3 = P(pdens3, vtot3)
E3 = E(vtot3, Bz3)

# Plot E and P to observe distributions

plt.plot(DateTime3, P3, label='P')
plt.plot(DateTime3, E3, label='E')
plt.xlabel('Date')
plt.ylabel('E (mV/m) & P (nPa)')
plt.grid()

plt.legend()

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
plt.xlabel('Date', fontsize=15)
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




############################################# Now compare all 3 spacecraft #########################################
#%% ################################################################################################################

plt.figure(figsize=(12, 6))
plt.grid()

# Adding labels and title
plt.xlabel('Date', fontsize=15)
plt.ylabel('SYM/H (nT)', fontsize=15)

# Set x-axis ticks
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Plot forecasted SYM/H in storm to compare to calculated SYM/H
plt.plot(DateTime1, sym_forecast3, label='ACE Forecasted SYM/H')
plt.plot(DateTime2, sym_forecast1, label='DSCOVR Forecasted SYM/H')
plt.plot(DateTime3, sym_forecast2, label='WIND Forecasted SYM/H')

plt.legend(fontsize=15)

plt.show()


#%% Plot small range to observe differences

downtoval = 103000
uptoval = 107000

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

plt.legend(fontsize=15)

plt.show()