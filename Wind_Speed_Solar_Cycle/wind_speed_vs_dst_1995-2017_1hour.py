# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:23:54 2023

@author: logan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read your data from CSV
df_wind_speed = pd.read_csv('wind_speed_1995-2023_1hour.csv')
df_dst = pd.read_csv('dst_1995-2017_1hour_unix.csv')

# Set the cutoff Unix timestamp at the end of the year 2016
cutoff_unix = pd.Timestamp('2016-12-31 23:00:00').timestamp()

# Filter the DataFrame to include only rows up to the end of 2016
df_wind_speed = df_wind_speed[df_wind_speed['Time'] <= cutoff_unix]


# Get the indices of the nan data points to be removed from df_wind_speed
wind_speed_indices_to_remove = df_wind_speed[df_wind_speed['WIND KP Speed, km/s'] == 9999].index
# Remove the same data points from df_dst as well as df_wind_speed
df_wind_speed = df_wind_speed.drop(wind_speed_indices_to_remove)
df_dst = df_dst.drop(wind_speed_indices_to_remove)


# Extract the needed data columns
wind_speed = df_wind_speed['WIND KP Speed, km/s']
dst = df_dst['Dst-index, nT']



#%%

# Create a scatter plot of the solar wind speed versus the Dst index
plt.figure(figsize=(10, 6))

plt.scatter(wind_speed, dst, color='black', marker='o', s=10)

# Set the axis labels
plt.xlabel('Solar wind speed (km/s)', fontsize=12)
plt.ylabel('Dst index (nT)', fontsize=12)

# Set the tick parameters
plt.tick_params(labelsize=10)

# Set the title
plt.title('Scatter Plot of Solar Wind Speed vs Dst Index (1995-2017)', fontsize=14)

# Show the plot
plt.show()

# Calculate the correlation coefficient between the solar wind speed and Dst index
correlation_coefficient = np.corrcoef(wind_speed, dst)[0][1]

print(f'Correlation coefficient: {correlation_coefficient}')

#%%

time = df_dst['Time']

plt.plot(time,wind_speed)
plt.plot(time,dst)
