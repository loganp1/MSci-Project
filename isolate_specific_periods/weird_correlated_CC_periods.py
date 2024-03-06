# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:33:37 2024

@author: logan
"""

import sys
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.style as style
from matplotlib.ticker import AutoMinorLocator

# Add path to the directory containing classes
sys.path.append(r"C:\Users\logan\OneDrive - Imperial College London\Uni\Year 4\MSci Project\MSci-Project"
                r"\Final_Model_OOP")

from Space_Weather_Forecasting_CLASS import Space_Weather_Forecast
from align_and_interpolate import align_and_interpolate_datasets
from SC_Propagation_CLASS import SC_Propagation

#%% Upload data

df_ACE = pd.read_csv('ace_data_unix.csv')
df_DSCOVR = pd.read_csv('dscovr_data_unix.csv')
df_Wind = pd.read_csv('wind_data_unix.csv')
df_SYM = pd.read_csv('SYM_data_unix.csv')
df_OMNI = pd.read_csv('OMNI_EP_data_unix.csv')

split_times = np.load('split_data_times_ALLf.npy')


#%% Create dictionary of data and initialise main class for forecasting

sc_dict = {'ACE': df_ACE, 'DSCOVR': df_DSCOVR, 'Wind': df_Wind}

MYclass = Space_Weather_Forecast(SC_dict=sc_dict, SYM_real=df_SYM, OMNI_data=df_OMNI)

#%% Split data into T hour periods

# DON'T RUN THIS AGAIN ONLY ONCE, RELOAD DATA IF DO ACCIDENTALLY
MYclass.SplitTimes(split_times, keep_primary_data=True)

#%% Filter to find our desired period index where we get the 

MYclass.SplitSubDFs(1000,1010)

#%%

SCsubDFs=MYclass.GetSCsubDFs()

#%% Get data

#zvCCsM, maxCCsM, deltaTsM = MYclass.GetCC(['multi','real'])
#zvCCsA, maxCCsA, deltaTsA = MYclass.GetCC(['ACE','real'])

#%%

SCsubDFs=MYclass.GetSCsubDFs()

total_nans = 0

count=0
for list_of_dfs in SCsubDFs:
    for df in list_of_dfs:
        count+=1
        total_nans += df.isna().sum().sum()
        print(count,'/6909')
        
#%%

max_consecutive_nans = 0
i=0
for list_of_dfs in SCsubDFs:
    for df in list_of_dfs:
        i+=1
        # Calculate consecutive NaN values in each column
        consecutive_nans = df.apply(lambda col: col.isna().astype(int).groupby(col.notna().astype(int).cumsum()).cumsum())
        
        # Get the maximum consecutive NaN value across all columns and update overall maximum
        max_consecutive_in_df = consecutive_nans.max().max()
        if max_consecutive_in_df > max_consecutive_nans:
            max_consecutive_nans = max_consecutive_in_df
        print(i)
print(f'The maximum number of consecutive NaN values in any column is: {max_consecutive_nans}')



#%% Plot parameters per SC

import matplotlib.dates as mdates

time_unix = SCsubDFs[0][0][0]['Time']
time = pd.to_datetime(time_unix,unit='s')

va = SCsubDFs[0][0][0]['v']
vd = SCsubDFs[1][0][0]['Speed, km/s']
vw = SCsubDFs[2][0][0]['KP_Speed, km/s']

plt.plot(time, va, label='ACE')
plt.plot(time, vd, label='DSCOVR')
plt.plot(time, vw, label='Wind')

# Set major locator to every hour
plt.gca().xaxis.set_major_locator(mdates.HourLocator())

# Set the time format for the x-axis ticks
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.title('Velocity')
plt.show()

#%%


na = SCsubDFs[0][0][0]['n']
nd = SCsubDFs[1][0][0]['Proton Density, n/cc']
nw = SCsubDFs[2][0][0]['Kp_proton Density, n/cc']

plt.plot(time, na, label='ACE')
plt.plot(time, nd, label='DSCOVR')
plt.plot(time, nw, label='Wind')

# Set major locator to every hour
plt.gca().xaxis.set_major_locator(mdates.HourLocator())

# Set the time format for the x-axis ticks
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.title('Density')
plt.show()

#%%


Bza = SCsubDFs[0][0][0]['Bz']
Bzd = SCsubDFs[1][0][0]['BZ, GSE, nT']
Bzw = SCsubDFs[2][0][0]['BZ, GSE, nT']

plt.plot(time, Bza, label='ACE')
plt.plot(time, Bzd, label='DSCOVR')
plt.plot(time, Bzw, label='Wind')

# Set major locator to every hour
plt.gca().xaxis.set_major_locator(mdates.HourLocator())

# Set the time format for the x-axis ticks
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.title('Bz')
plt.show()

#%%


Ea = SCsubDFs[0][0]['E']
Ed = SCsubDFs[1][0]['E']
Ew = SCsubDFs[2][0]['E']

plt.plot(time, Ea, label='ACE')
plt.plot(time, Ed, label='DSCOVR')
plt.plot(time, Ew, label='Wind')

# Set major locator to every hour
plt.gca().xaxis.set_major_locator(mdates.HourLocator())

# Set the time format for the x-axis ticks
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.title('E')
plt.show()


Pa = SCsubDFs[0][0]['P']
Pd = SCsubDFs[1][0]['P']
Pw = SCsubDFs[2][0]['P']

plt.plot(time, Pa, label='ACE')
plt.plot(time, Pd, label='DSCOVR')
plt.plot(time, Pw, label='Wind')

# Set major locator to every hour
plt.gca().xaxis.set_major_locator(mdates.HourLocator())

# Set the time format for the x-axis ticks
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.title('P')
plt.show()
