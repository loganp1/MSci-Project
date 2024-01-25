# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 06:21:23 2023

@author: logan
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load and prepare DataFrame
df = pd.read_csv('wind_1995-2023_1day.csv')

# Create a time series from the Year, Day, and Hour columns
df['Time'] = pd.to_datetime(df['Year'].astype(str) + df['Day'].astype(str) + df['Hour'].astype(str), 
                            format='%Y%j%H', errors='coerce')

# Remove rows with magnetic field data values of 999.9
df = df[df['WIND Bx_gse, nT'] != 999.9]
# Don't need the following 2 as data issues occur for all components simultaneously
#df = df[df['WIND By_gse, nT'] != 999.9]
#df = df[df['WIND Bz_gse, nT'] != 999.9]

# Define Solar Cycle start dates
solar_cycle_starts = [1996, 2008, 2019]
solar_cycle_number = [23, 24, 25]

time = df['Time']

#%%

# Plot the time series vs. the Bx column data
plt.figure(figsize=(12, 6))
plt.plot(time, df['WIND Bx_gse, nT'], label='Bx')
plt.xlabel('Time')
plt.ylabel('WIND Bx_gse, nT')
#plt.title('Magnetic Field vs. Time')
plt.grid()

# Label every 5 years on the x-axis
years = range(df['Year'].min(), df['Year'].max() + 1, 5)
plt.xticks([pd.to_datetime(f'{year}-01-01') for year in years])

# Add vertical lines to distinguish Solar Cycles
for year in solar_cycle_starts:
    solar_cycle_date = pd.to_datetime(f'{year}-08-01')  # Assuming August as the starting month for Solar Cycle
    plt.axvline(solar_cycle_date, color='red', linestyle='--', 
                label=f'Solar Cycle {solar_cycle_number[solar_cycle_starts.index(year)]}')
    
plt.legend()
plt.show()

#%%

# Plot the time series vs. the By column data
plt.figure(figsize=(12, 6))
plt.plot(df['Time'], df['WIND By_gse, nT'], label='By')
plt.xlabel('Time')
plt.ylabel('WIND By_gse, nT')
#plt.title('Magnetic Field vs. Time')
plt.grid()

# Label every 5 years on the x-axis
years = range(df['Year'].min(), df['Year'].max() + 1, 5)
plt.xticks([pd.to_datetime(f'{year}-01-01') for year in years])

# Add vertical lines to distinguish Solar Cycles
for year in solar_cycle_starts:
    solar_cycle_date = pd.to_datetime(f'{year}-08-01')  # Assuming August as the starting month for Solar Cycle
    plt.axvline(solar_cycle_date, color='red', linestyle='--', 
                label=f'Solar Cycle {solar_cycle_number[solar_cycle_starts.index(year)]}')

plt.legend(loc='upper right')
plt.show()

#%%

# Plot the time series vs. the Bz column data
plt.figure(figsize=(12, 6))
plt.plot(df['Time'], df['WIND Bz_gse, nT'], label='Bz')
plt.xlabel('Time')
plt.ylabel('WIND Bz_gse, nT')
#plt.title('Magnetic Field vs. Time')
plt.grid()

# Label every 5 years on the x-axis
years = range(df['Year'].min(), df['Year'].max() + 1, 5)
plt.xticks([pd.to_datetime(f'{year}-01-01') for year in years])

# Add vertical lines to distinguish Solar Cycles
for year in solar_cycle_starts:
    solar_cycle_date = pd.to_datetime(f'{year}-08-01')  # Assuming August as the starting month for Solar Cycle
    plt.axvline(solar_cycle_date, color='red', linestyle='--', 
                label=f'Solar Cycle {solar_cycle_number[solar_cycle_starts.index(year)]}')

plt.legend()
plt.show()

