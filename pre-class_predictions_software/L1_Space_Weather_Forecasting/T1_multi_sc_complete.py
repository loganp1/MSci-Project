# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:46:28 2024

@author: logan
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from forecasting_model import SYM_forecast
from scipy.optimize import curve_fit
from weightedAv_propagator_function import EP_weightedAv_propagator
from singleSC_propagator_function import EP_singleSC_propagator

#%%

# Import actual SYM/H data

df_sym = pd.read_csv('SYM1_unix.csv')

df_sym['DateTime'] = pd.to_datetime(df_sym['Time'], unit='s')

df_sym = df_sym[541280:] # Do this as only acceptable data for wind in T1 is past here

DateTime = df_sym['DateTime'].values
sym = df_sym['SYM/H, nT'].values

plt.plot(DateTime,sym)

#%%

##################################################### MULTI-SPACECRAFT #############################################
####################################################################################################################

# Read in B field and plasma data for the DSCOVR dataset
df_params1 = pd.read_csv('dscovr_T1_1min_unix.csv')
df_params1 = df_params1[541280:]

df_params2 = pd.read_csv('wind_T1_1min_unix.csv')
df_params2 = df_params2[541280:]

df_params3 = pd.read_csv('ace_T1_1min_unix.csv')
df_params3 = df_params3[541280:]


#%% Now here's where it gets AMAZING!!! Apply my weighted average function which should do all the cleaning
### and return the propagated E and P distributions!

[E_prop,time],[P_prop,time] = EP_weightedAv_propagator(df_params1,df_params2,df_params3)
DateTime_prop = pd.to_datetime(time, unit='s')


#%% Use model to forecast SYM/H, starting with a basic model without propagation

# Extract initial SYM/H value for model

sym_forecast = []
initial_sym = sym[0]  # Record this for later, as current_sym will change
current_sym = sym[0]

for i in range(len(DateTime_prop)-1):
    
    new_sym = SYM_forecast(current_sym,
                                     P_prop[i],
                                     P_prop[i+1],
                                     E_prop[i])
    sym_forecast.append(new_sym)
    current_sym = new_sym
    

sym_forecast.insert(0,initial_sym)   # Add initial value we used to propagate through forecast


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
plt.plot(DateTime_prop, sym_forecast, label = 'DSCOVR Forecasted SYM/H')

plt.legend(loc='upper left',fontsize=15)
plt.show()



############################################# Now compare to single spacecraft #####################################
#%% ################################################################################################################

# Forecast using ACE as just discovered it gives same result as MultiSC in my classes
time3, E3, P3, dTs3 = EP_singleSC_propagator(df_params3,'ACE')
DateTime3 = pd.to_datetime(time3, unit='s')

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

#%%

plt.figure(figsize=(12, 6))
plt.grid()

# Adding labels and title
plt.xlabel('Day', fontsize=15)
plt.ylabel('SYM/H (nT)', fontsize=15)

# Set x-axis ticks
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Plot forecasted SYM/H in storm to compare to calculated SYM/H
plt.plot(DateTime, sym_forecast, label='Multi-Spacecraft Weighted Average Forecasted SYM/H')
plt.plot(DateTime3, sym_forecast3)

plt.legend(fontsize=15)

plt.show()

# EXCELLENT THEYRE DIFFERENT! Must have a bug in my classes probably plotting same data


#%% Perform cross-correlation analysis 

from cross_correlation import cross_correlation

lags = np.arange(-len(sym_forecast) + 1, len(sym_forecast))
dt = 60

# Calculate corresponding time delays
time_delays = lags * dt


