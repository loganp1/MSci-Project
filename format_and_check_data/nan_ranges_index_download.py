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
max_allowed_nans_dict = {'n': 10, 'vx': 10, 'Bz': 10}

# Loop through selected columns and find good data regions
for column_name in selected_columns:
    max_allowed_nans = max_allowed_nans_dict.get(column_name, 10)
    start_indices, end_indices = find_good_regions(column_name, max_allowed_nans)
    good_data_dict[column_name] = np.array([[df['Time'][start], df['Time'][end]] for start, end in zip(start_indices, 
                                                                                                       end_indices)])

# Save the good data dictionary to a file - no we need to save the single array below
# np.save('ace_nNaNs_max10_times_new.npy', good_data_dict)

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

np.save('ace_nNaNs_max10_times_new.npy', good_periods)


#%% Need to use for 9 arrays - 3 for each spacecraft

def find_common_intervals(*arrays):
    result = []
    
    pointers = [0] * len(arrays)
    
    while all(pointers[i] < len(arr) for i, arr in enumerate(arrays)):
        start_time = max(arr[pointers[i]][0] for i, arr in enumerate(arrays))
        end_time = min(arr[pointers[i]][1] for i, arr in enumerate(arrays))
        
        if start_time <= end_time:
            result.append([start_time, end_time])
        
        # Move pointers based on the smallest end time
        min_end_time = min(arr[pointers[i]][1] for i, arr in enumerate(arrays))
        for i, arr in enumerate(arrays):
            if arr[pointers[i]][1] == min_end_time:
                pointers[i] += 1
    
    return result

#%%

df_ACE = pd.read_csv('ace_data_unix.csv')
df_DSCOVR = pd.read_csv('dscovr_data_unix.csv')
df_Wind = pd.read_csv('wind_data_unix.csv')

#%%

##### YOU NEED TO APPLY THE find_good_regions to ALL OF THESE THEN INPUT THE GOOD_PERIODS INTO THE FOLLOWING FN

ALL3_good_periods = find_common_intervals(df_ACE['n'], df_ACE['vx'], df_ACE['Bz'], 
                                          df_DSCOVR['Proton Density, n/cc'], df_DSCOVR['Vx Velocity,km/s'], 
                                          df_DSCOVR['BZ, GSE, nT'], 
                                          df_Wind['Kp_proton Density, n/cc'], df_Wind['KP_Vx,km/s'], 
                                          df_Wind['BZ, GSE, nT'])

#%%

# Specify the column names you want to consider - use any of 3 velocity comps as all match up in terms of nans
#selected_columns = ['n', 'vx', 'Bz']

# Function to find good data regions for a specific column
def find_good_regions(df, column_name, max_allowed_nans):
    nan_mask = df[column_name].isnull()
    consecutive_nans = nan_mask.groupby((~nan_mask).cumsum()).cumsum()
    good_regions = consecutive_nans <= max_allowed_nans
    start_indices = good_regions.index[good_regions & ~good_regions.shift(fill_value=False)].tolist()
    end_indices = good_regions.index[~good_regions & good_regions.shift(fill_value=False)].tolist()
    return start_indices, end_indices

# Dictionary to store the good data for each column
ACE_good_data_dict = {}
DSC_good_data_dict = {}
Wind_good_data_dict = {}

#%%

# DONT FORGET WE NEED TO MAKE NAN VALUES FOR SECOND 2

Re = 6378

# Replace 'DSCOVR' with 'df_DSCOVR'
df_DSCOVR['Field Magnitude,nT'] = df_DSCOVR['Field Magnitude,nT'].replace(9999.99, np.nan)
df_DSCOVR['Vector Mag.,nT'] = df_DSCOVR['Vector Mag.,nT'].replace(9999.99, np.nan)
df_DSCOVR['BX, GSE, nT'] = df_DSCOVR['BX, GSE, nT'].replace(9999.99, np.nan)
df_DSCOVR['BY, GSE, nT'] = df_DSCOVR['BY, GSE, nT'].replace(9999.99, np.nan)
df_DSCOVR['BZ, GSE, nT'] = df_DSCOVR['BZ, GSE, nT'].replace(9999.99, np.nan)
df_DSCOVR['Speed, km/s'] = df_DSCOVR['Speed, km/s'].replace(99999.9, np.nan)
df_DSCOVR['Vx Velocity,km/s'] = df_DSCOVR['Vx Velocity,km/s'].replace(99999.9, np.nan)
df_DSCOVR['Vy Velocity, km/s'] = df_DSCOVR['Vy Velocity, km/s'].replace(99999.9, np.nan)
df_DSCOVR['Vz Velocity, km/s'] = df_DSCOVR['Vz Velocity, km/s'].replace(99999.9, np.nan)
df_DSCOVR['Proton Density, n/cc'] = df_DSCOVR['Proton Density, n/cc'].replace(999.999, np.nan)
df_DSCOVR['Wind, Xgse,Re'] = df_DSCOVR['Wind, Xgse,Re'].replace(9999.99, np.nan)
df_DSCOVR['Wind, Ygse,Re'] = df_DSCOVR['Wind, Ygse,Re'].replace(9999.99, np.nan)
df_DSCOVR['Wind, Zgse,Re'] = df_DSCOVR['Wind, Zgse,Re'].replace(9999.99, np.nan)

# Have to do change to km AFTER removing vals else fill value will change!
df_DSCOVR['Wind, Xgse,Re'] = df_DSCOVR['Wind, Xgse,Re'] * Re
df_DSCOVR['Wind, Ygse,Re'] = df_DSCOVR['Wind, Ygse,Re'] * Re
df_DSCOVR['Wind, Zgse,Re'] = df_DSCOVR['Wind, Zgse,Re'] * Re


# Replace 'Wind' with 'df_Wind'
df_Wind['BX, GSE, nT'] = df_Wind['BX, GSE, nT'].replace(9999.990000, np.nan)
df_Wind['BY, GSE, nT'] = df_Wind['BY, GSE, nT'].replace(9999.990000, np.nan)
df_Wind['BZ, GSE, nT'] = df_Wind['BZ, GSE, nT'].replace(9999.990000, np.nan)
df_Wind['Vector Mag.,nT'] = df_Wind['Vector Mag.,nT'].replace(9999.990000, np.nan)
df_Wind['Field Magnitude,nT'] = df_Wind['Field Magnitude,nT'].replace(9999.990000, np.nan)
df_Wind['KP_Vx,km/s'] = df_Wind['KP_Vx,km/s'].replace(99999.900000, np.nan)
df_Wind['Kp_Vy, km/s'] = df_Wind['Kp_Vy, km/s'].replace(99999.900000, np.nan)
df_Wind['KP_Vz, km/s'] = df_Wind['KP_Vz, km/s'].replace(99999.900000, np.nan)
df_Wind['KP_Speed, km/s'] = df_Wind['KP_Speed, km/s'].replace(99999.900000, np.nan)
df_Wind['Kp_proton Density, n/cc'] = df_Wind['Kp_proton Density, n/cc'].replace(999.990000, np.nan)
df_Wind['Wind, Ygse,Re'] = df_Wind['Wind, Ygse,Re'].replace(9999.990000, np.nan)
df_Wind['Wind, Zgse,Re'] = df_Wind['Wind, Zgse,Re'].replace(9999.990000, np.nan)

df_Wind['Wind, Xgse,Re'] = df_Wind['Wind, Xgse,Re']*Re
df_Wind['Wind, Ygse,Re'] = df_Wind['Wind, Ygse,Re']*Re
df_Wind['Wind, Zgse,Re'] = df_Wind['Wind, Zgse,Re']*Re


# Loop through selected columns and find good data regions
#%%
dfcount = 1
for df in [df_ACE,df_DSCOVR,df_Wind]:
    if dfcount == 1:
        columns = ['n','vx','Bz']
    if dfcount == 2:
        columns = ['Proton Density, n/cc','Vx Velocity,km/s','BZ, GSE, nT']
    if dfcount == 3:
        columns = ['Kp_proton Density, n/cc','KP_Vx,km/s','BZ, GSE, nT']
    for column_name in columns:
        max_allowed_nans = 10
        start_indices, end_indices = find_good_regions(df, column_name, max_allowed_nans)
        if dfcount == 1:
            ACE_good_data_dict[column_name] = np.array([[df['Time'][start], df['Time'][end]] for start, 
                                                    end in zip(start_indices, end_indices)])
        if dfcount == 2:
            DSC_good_data_dict[column_name] = np.array([[df['Time'][start], df['Time'][end]] for start, 
                                                    end in zip(start_indices, end_indices)])
        if dfcount == 3:
            Wind_good_data_dict[column_name] = np.array([[df['Time'][start], df['Time'][end]] for start, 
                                                    end in zip(start_indices, end_indices)])
    dfcount += 1

#%% Great, now apply these to the find_common_intervals function

ALL3_good_periods = find_common_intervals(ACE_good_data_dict['n'],ACE_good_data_dict['vx'],ACE_good_data_dict['Bz'],
                                          DSC_good_data_dict['Proton Density, n/cc'],
                                          DSC_good_data_dict['Vx Velocity,km/s'],
                                          DSC_good_data_dict['BZ, GSE, nT'],
                                          Wind_good_data_dict['Kp_proton Density, n/cc'], 
                                          Wind_good_data_dict['KP_Vx,km/s'], 
                                          Wind_good_data_dict['BZ, GSE, nT'])

ALL3_good_periods = np.asarray(ALL3_good_periods)
                                          
#%%

np.save('max10_NaNs_ALLdata_start_end_times.npy', ALL3_good_periods)