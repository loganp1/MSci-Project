# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 15:20:35 2024

@author: logan
"""

# Import modules

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import seaborn as sns

#%% Load appropriate data - multi predictions and real SYM

# CCs
CCs = np.load('MvsR_maxCCs_improved.npy')

# Time shifts zero-val to max CC
dTs = np.load("MvsR_deltaTs_improved.npy")

# Split up time periods
period_times = np.load("split_data_times.npy")

# SYM predictions
multi_sym = np.load('multi_sym_forecastnpy.npy')

# Real SYM/H
real_sym = pd.read_csv("SYM_data_unix.csv")

#%% Create function to calculate displacement integral

displacement = []
for i in range(len(period_times)):
    print(i)
    symFilt = multi_sym[1][(multi_sym[0] >= period_times[i][0]) & (multi_sym[0] <= period_times[i][1])]
    displacement.append(abs(sum(sym for sym in symFilt if sym<0)))
    
    
#%% Displacement vs CC

# Create bins for the x-axis (displacement)
bins = np.linspace(min(displacement), max(displacement), num=100)  # Adjust number of bins as needed

# Use pandas to group the data by the bins and calculate the mean of 'CCs' in each bin
df = pd.DataFrame({'displacement': displacement, 'CCs': CCs})
df['displacement_bin'] = pd.cut(df['displacement'], bins=bins)
averages = df.groupby('displacement_bin')['CCs'].mean()

# Calculate the count of data points in each bin
bin_counts = df['displacement_bin'].value_counts()

# Set the minimum number of data points in a bin
min_data_points = 20

# Filter out bins that don't meet the minimum data points criterion
valid_bins = bin_counts[bin_counts >= min_data_points].index
filtered_averages = averages.loc[valid_bins]

### IMPORTANT: We have set min_data_points = 51 to remove a somewhat outlier which appears with 50 bins
###            I will also remove the first bin here which seems to be an outlier (as we'd expect as low
###            sym is just noise!!!)

# Sort the bins by their midpoints
sorted_bins = filtered_averages.index.sort_values()
filtered_averages = filtered_averages.loc[sorted_bins]

filtered_averages = filtered_averages.iloc[1:]

# Calculate midpoints of the bins for the filtered data
bin_midpoints = filtered_averages.index.map(lambda x: x.mid).astype(float)

# Plot the binned data
with plt.style.context('ggplot'):
    sns.set_palette("tab10")
    plt.scatter(bin_midpoints[:-1], filtered_averages.values[:-1], marker='x', label=f'Bin Mean (20 minimum data pts)')
                                                

# Fit a line of best fit using numpy.polyfit
degree = 1  # Linear fit
coefficients = np.polyfit(bin_midpoints[:-1], filtered_averages.values[:-1], degree)
line_of_best_fit = np.polyval(coefficients, bin_midpoints)

# Plot the line of best fit
plt.plot(bin_midpoints, line_of_best_fit, color='red', label='Linear Regression Line of Best Fit')

plt.xlabel('Displacement (Cumulative SYM/H)',fontsize=15)
plt.ylabel('Cross-Correlation',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='upper left',fontsize=10)
plt.show()


#%% Displacement vs Time Offset
    
# Create bins for the x-axis (displacement)
bins = np.linspace(min(displacement), max(displacement), num=100)  # You can adjust the number of bins as needed

# Use pandas to group the data by the bins and calculate the mean of 'CCs' in each bin
df = pd.DataFrame({'displacement': displacement, 'dTs': dTs})
df['displacement_bin'] = pd.cut(df['displacement'], bins=bins)
df = df[(df['dTs'] >= 0) & (df['dTs'] <= 3600)] # Filter the values to keep only time offsets between 0 & 60 mins
averages = df.groupby('displacement_bin')['dTs'].mean()

# Calculate the count of data points in each bin
bin_counts = df['displacement_bin'].value_counts()

# Set the minimum number of data points in a bin
min_data_points = 10  # You can adjust this value as needed

# Filter out bins that don't meet the minimum data points criterion
valid_bins = bin_counts[bin_counts >= min_data_points].index
filtered_averages = averages.loc[valid_bins]

# Calculate midpoints of the bins for the filtered data
bin_midpoints = filtered_averages.index.map(lambda x: x.mid).astype(float)

# Plot the binned data
plt.scatter(bin_midpoints, filtered_averages.values, marker='x', label=f'Bin Mean with {min_data_points}'
                                                                        f' Min. Data Points: ')
                                                

# Fit a line of best fit using numpy.polyfit
# degree = 1  # Linear fit
# coefficients = np.polyfit(bin_midpoints, filtered_averages.values, degree)
# line_of_best_fit = np.polyval(coefficients, bin_midpoints)

# Plot the line of best fit
#plt.plot(bin_midpoints, line_of_best_fit, color='red', label='Line of Best Fit')

plt.xlabel('Displacement')
plt.ylabel('dTs')
plt.legend()
plt.show()
