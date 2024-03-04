# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:17:44 2024

@author: logan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from filter_good_data import modify_list

df = pd.read_csv('ace_data_unix.csv')

#%%

nan_indices = df[df['n'].isnull()].index.tolist()

#print(nan_indices)

#%% Plot time vs n to see timescale of variations and what sort of range interpolation may be appr. for

plt.plot(df['Time'],df['n'])
plt.show()

# Interpolate linearly
df_interp = df.interpolate()

# Replot
plt.plot(df_interp['Time'],df_interp['n'])
plt.show()

#%%

# Now let's identify some good regions of data with a maximum consecutive allowed nan value
column_name = 'n'
max_allowed_nans = 10  # You can change this value as needed

# Find consecutive NaN values using a rolling window
nan_mask = df[column_name].isnull()
consecutive_nans = nan_mask.groupby((~nan_mask).cumsum()).cumsum()

# Identify regions with no more than max_allowed_nans consecutive NaNs
good_regions = consecutive_nans <= max_allowed_nans

# Get the indices of the start and end of each good region
start_indices = good_regions.index[good_regions & ~good_regions.shift(fill_value=False)].tolist()
end_indices = good_regions.index[~good_regions & good_regions.shift(fill_value=False)].tolist()

# Display the identified regions
for start, end in zip(start_indices, end_indices):
    print(f"Start Index: {start}, End Index: {end}")

# If there are no good regions, print a message
if not start_indices:
    print(f"No regions found with less than or equal to {max_allowed_nans} consecutive NaNs.")

gaps = []

for i in range(len(start_indices)-1):
    gaps.append(np.asarray(start_indices[i+1]) - np.asarray(end_indices[i]))
    
    
#%%

gaps_in_hours = np.asarray(gaps)/60

#%% Plot GOOD ranges 

gdata_inds = np.asarray(end_indices)-np.asarray(start_indices)

plt.hist(gdata_inds,bins=1000);

#%% Create array in required form for Ned's filter_good_data function

good_data = np.array([[df['Time'][start], df['Time'][end]] for start, end in zip(start_indices, end_indices)])

#np.save('ace_nNaNs_max10_times.npy', good_data)



#%% From ChatGPT: A function to apply the above to multiple columns

df = pd.read_csv('ace_data_unix.csv')

#%%

# Specify the column names you want to consider - use any of 3 velocity comps as all match up in terms of nans
selected_columns = ['n', 'vx', 'Bz']

# Function to find good data regions for a specific column
def find_good_regions(column_name, max_allowed_nans):
    nan_mask = df[column_name].isnull()
    consecutive_nans = nan_mask.groupby((~nan_mask).cumsum()).cumsum()
    good_regions = consecutive_nans <= max_allowed_nans
    start_indices = good_regions.index[good_regions & ~good_regions.shift(fill_value=False)].tolist()
    end_indices = good_regions.index[~good_regions & good_regions.shift(fill_value=False)].tolist()
    return start_indices, end_indices

# Dictionary to store the good data for each column
good_data_dict = {}

# Specify the maximum allowed consecutive NaNs for each column
max_allowed_nans_dict = {'n': 10, 'other_column1': 10, 'other_column2': 10}

# Loop through selected columns and find good data regions
for column_name in selected_columns:
    max_allowed_nans = max_allowed_nans_dict.get(column_name, 10)
    start_indices, end_indices = find_good_regions(column_name, max_allowed_nans)
    good_data_dict[column_name] = np.array([[df['Time'][start], df['Time'][end]] for start, end in zip(start_indices, 
                                                                                                       end_indices)])

# Save the good data dictionary to a file
#np.save('ace_nNaNs_max10_times_dict.npy', good_data_dict)

#%%

def find_common_intervals(arr1, arr2, arr3):
    result = []
    
    i, j, k = 0, 0, 0
    
    while i < len(arr1) and j < len(arr2) and k < len(arr3):
        start_time = max(arr1[i][0], arr2[j][0], arr3[k][0])
        end_time = min(arr1[i][1], arr2[j][1], arr3[k][1])
        
        if start_time <= end_time:
            result.append([start_time, end_time])
        
        # Move pointers based on the smallest end time
        min_end_time = min(arr1[i][1], arr2[j][1], arr3[k][1])
        if arr1[i][1] == min_end_time:
            i += 1
        if arr2[j][1] == min_end_time:
            j += 1
        if arr3[k][1] == min_end_time:
            k += 1
    
    return result

# Example usage:
arr1 = [[1, 5], [6, 10], [12, 15]]
arr2 = [[2, 7], [8, 9], [14, 18]]
arr3 = [[3, 6], [9, 12], [13, 16]]

result = find_common_intervals(arr1, arr2, arr3)
print(result)

# Think this sort of works but returns same start and end times as well - no worries can remove these

# Apply to our data

good_periods = np.asarray(find_common_intervals(good_data_dict['n'],good_data_dict['vx'],good_data_dict['Bz']))

good_periods_test=good_periods==good_data
