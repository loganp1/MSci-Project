# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:39:32 2023

@author: logan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read your data from CSV
df_wind_2000 = pd.read_csv('wind_2000_1min_unix.csv')

# Clean Data
df_wind_2000 = df_wind_2000[df_wind_2000['Flow speed, km/s'] != 99999.9]

# Extract the speed and time data
wind_speed = df_wind_2000['Flow speed, km/s']
wind_time = df_wind_2000['Time']

# Set the number of bins for the x and y axes
num_bins_x = 50  # Number of bins for the time axis (x-axis)
num_bins_y = 50  # Number of bins for the speed axis (y-axis)

# Create the 2D histogram using plt.hist2d
hist, xedges, yedges = np.histogram2d(wind_time, wind_speed, bins=[num_bins_x, num_bins_y])

# Create a contour plot to represent the frequency (color scheme)
plt.contourf(xedges[:-1], yedges[:-1], hist.T, cmap='coolwarm')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Wind Speed (km/s)')
plt.title('Year 2000 1min data')
plt.ylim(min(wind_speed),750)

# Show the plot
plt.colorbar(label='Frequency')
plt.show()


#%%

# Repeat for different time lengths

tslice = 10000

# Extract the speed and time data
wind_speed2 = df_wind_2000['Flow speed, km/s'][:tslice]
wind_time2 = df_wind_2000['Time'][:tslice]

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
#plt.title('1995-1996')
plt.ylim(min(wind_speed2),750)

# Show the plot
plt.colorbar(label='Frequency')
plt.show()