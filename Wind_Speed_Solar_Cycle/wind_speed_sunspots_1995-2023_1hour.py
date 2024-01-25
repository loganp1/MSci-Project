# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:01:39 2023

@author: logan
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.colors import PowerNorm
import numpy as np
from datetime import datetime, timedelta

# Read your data from CSV
df_wind_speed = pd.read_csv('wind_speed_1995-2023_1hour.csv')
df_sunspots = pd.read_csv('sunspots_1995-2023_monthly_average.csv')

# Drop final row to better match wind speed data
df_sunspots = df_sunspots.drop(df_sunspots.index[-8:])

#%%

# Function for converting 'year_val' column to unix timestamp
def float_year_to_unix_timestamp(year_fraction):
    year = int(year_fraction)
    fraction_of_year = year_fraction - year
    base_date = datetime(year, 1, 1)
    delta_days = int(fraction_of_year * 365)  # Assuming a non-leap year
    target_date = base_date + timedelta(days=delta_days)
    timestamp = target_date.timestamp()
    return int(timestamp)

# Example usage
year_fraction = 2000.01
unix_timestamp = float_year_to_unix_timestamp(year_fraction)
print(f"Unix timestamp for {year_fraction} is: {unix_timestamp}")

#%%

# Choose type of histogram you want

type_hist = 'log'
#type_hist = 'linear'

# Clean Data
df_wind_speed = df_wind_speed[df_wind_speed['WIND KP Speed, km/s'] != 9999]

# Convert df_sunspots time into unix with new column 'Time':
df_sunspots['Time'] = df_sunspots['year_val'].apply(float_year_to_unix_timestamp)

# Extract the speed and time data
wind_speed = df_wind_speed['WIND KP Speed, km/s']
wind_time = df_wind_speed['Time']

# Set the number of bins for the x and y axes
num_bins_x = 50  # Number of bins for the time axis (x-axis)
num_bins_y = 50  # Number of bins for the speed axis (y-axis)

# Create the 2D histogram using plt.hist2d
hist, xedges, yedges = np.histogram2d(wind_time, wind_speed, bins=[num_bins_x, num_bins_y])

# Replace all 0s in hist with 1s so that can do logarithmic plot and 0s wont be ignored as white space
hist[hist == 0] = 1

# Create a 2D image plot with continuous color bar
non_zero_values = hist[hist > 0]
min_non_zero = np.min(non_zero_values)

if type_hist == 'log':
    plt.imshow(hist.T, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()], 
               aspect='auto', cmap='coolwarm', origin='lower', norm=LogNorm())
    
if type_hist == 'linear':
    plt.imshow(hist.T, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()], 
               aspect='auto', cmap='coolwarm', origin='lower', norm=Normalize(vmin=np.min(hist), vmax=np.max(hist)))

else:
    plt.imshow(hist.T, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()], 
           aspect='auto', cmap='coolwarm', origin='lower', norm=PowerNorm(gamma=0.5, vmin=np.min(hist), vmax=np.max(hist)))


# Add labels and title
plt.xlabel('Time')
plt.ylabel('Wind Speed (km/s)')

# Create a secondary y-axis for sunspot data
ax2 = plt.gca().twinx()

# Plot the sunspot data on the secondary y-axis
ax2.plot(df_sunspots['Time'], df_sunspots['mean'], color='black', label='Sunspot Number')

# Set y-axis limits for the histogram
plt.ylim(min(wind_speed), 800)

# Set y-axis limits for the sunspot data
ax2.set_ylim(bottom=df_sunspots['mean'].min(), top=df_sunspots['mean'].max())

# Set the secondary y-axis label
ax2.set_ylabel('Sunspot Number', color='black')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
cbar = plt.colorbar(label='Frequency')
cbar.ax.set_position([0.85, 0.2, 0.4, 0.7])  # Adjust the position of the colorbar

# Set the colorbar label
cbar.set_label('Frequency')

plt.legend()
plt.show()


#%%

# Repeat for different time lengths

tslice = 10000

# Extract the speed and time data
wind_speed2 = df_wind_speed['WIND KP Speed, km/s'][:tslice]
wind_time2 = df_wind_speed['Time'][:tslice]

# Set the number of bins for the x and y axes
num_bins_x = 50  # Number of bins for the time axis (x-axis)
num_bins_y = 50  # Number of bins for the speed axis (y-axis)

# Create the 2D histogram using plt.hist2d
hist, xedges, yedges = np.histogram2d(wind_time2, wind_speed2, bins=[num_bins_x, num_bins_y])

# Create a contour plot to represent the frequency (color scheme)
plt.contourf(xedges[:-1], yedges[:-1], hist.T, cmap='coolwarm')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Wind Speed (km/s)')
plt.title('1995-1996 1hour data')
plt.ylim(min(wind_speed2),750)

# Show the plot
plt.colorbar(label='Frequency')
plt.show()

