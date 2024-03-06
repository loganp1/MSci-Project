# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 23:23:53 2024

@author: logan
"""

import pandas as pd
import numpy as np

DSCOVR = pd.read_csv('dscovr_data_unix.csv')

Re=6378
DSCOVR['Field Magnitude,nT'] = DSCOVR['Field Magnitude,nT'].replace(9999.99, np.nan)
DSCOVR['Vector Mag.,nT'] = DSCOVR['Vector Mag.,nT'].replace(9999.99, np.nan)
DSCOVR['BX, GSE, nT'] = DSCOVR['BX, GSE, nT'].replace(9999.99, np.nan)
DSCOVR['BY, GSE, nT'] = DSCOVR['BY, GSE, nT'].replace(9999.99, np.nan)
DSCOVR['BZ, GSE, nT'] = DSCOVR['BZ, GSE, nT'].replace(9999.99, np.nan)
DSCOVR['Speed, km/s'] = DSCOVR['Speed, km/s'].replace(99999.9, np.nan)
DSCOVR['Vx Velocity,km/s'] = DSCOVR['Vx Velocity,km/s'].replace(99999.9, np.nan)
DSCOVR['Vy Velocity, km/s'] = DSCOVR['Vy Velocity, km/s'].replace(99999.9, np.nan)
DSCOVR['Vz Velocity, km/s'] = DSCOVR['Vz Velocity, km/s'].replace(99999.9, np.nan)
DSCOVR['Proton Density, n/cc'] = DSCOVR['Proton Density, n/cc'].replace(999.999, np.nan)
DSCOVR['Wind, Xgse,Re'] = DSCOVR['Wind, Xgse,Re'].replace(9999.99, np.nan)
DSCOVR['Wind, Ygse,Re'] = DSCOVR['Wind, Ygse,Re'].replace(9999.99, np.nan)
DSCOVR['Wind, Zgse,Re'] = DSCOVR['Wind, Zgse,Re'].replace(9999.99, np.nan)

# Have to do change to km AFTER removing vals else fill value will change!
DSCOVR['Wind, Xgse,Re'] = DSCOVR['Wind, Xgse,Re']*Re
DSCOVR['Wind, Ygse,Re'] = DSCOVR['Wind, Ygse,Re']*Re
DSCOVR['Wind, Zgse,Re'] = DSCOVR['Wind, Zgse,Re']*Re

#%%

nan_count = DSCOVR['Proton Density, n/cc'].isna().sum()
print(f"Number of NaN values in 'vx' column: {nan_count}")
print(f"Number of non-NaN values in 'vx' column: {len(DSCOVR['Vx Velocity,km/s'])-nan_count}")

nan_count = DSCOVR['BZ, GSE, nT'].isna().sum()
print(f"Number of NaN values in 'Bz' column: {nan_count}")
print(f"Number of non-NaN values in 'Bz' column: {len(DSCOVR['BZ, GSE, nT'])-nan_count}")

#%%

# Replace column names and DataFrame name accordingly
nan_indices_density = DSCOVR[DSCOVR['Proton Density, n/cc'].isna()].index
nan_indices_bz = DSCOVR[DSCOVR['BZ, GSE, nT'].isna()].index
nan_indices_velocity = DSCOVR[DSCOVR['Vx Velocity,km/s'].isna()].index

print("Indices of NaN values in 'Proton Density, n/cc':", nan_indices_density)
print("Indices of NaN values in 'BZ, GSE, nT':", nan_indices_bz)
print("Indices of NaN values in 'Vx Velocity,km/s':", nan_indices_velocity)

#%%

dsc_nan_inds = []

columns_to_check = ['Proton Density, n/cc', 'BZ, GSE, nT', 'Vx Velocity,km/s']

for column in columns_to_check:
    nan_indices = DSCOVR[DSCOVR[column].isna()].index
    dsc_nan_inds.append(nan_indices)
    
    successive_nan_lengths = []
    current_length = 0
    
    for i in range(1, len(nan_indices)):
        if nan_indices[i] == nan_indices[i - 1] + 1:
            current_length += 1
        else:
            if current_length > 0:
                successive_nan_lengths.append(current_length + 1)
                current_length = 0

    if current_length > 0:
        successive_nan_lengths.append(current_length + 1)

    #dsc_nan_inds.append(successive_nan_lengths)
    
    
#%%

dsc_n_nans, dsc_Bz_nans, dsc_v_nans = dsc_nan_inds


#%%

Wind = pd.read_csv('Wind_data_unix.csv')

Re=6378
Wind['BX, GSE, nT'] = Wind['BX, GSE, nT'].replace(9999.990000, np.nan)
Wind['BY, GSE, nT'] = Wind['BY, GSE, nT'].replace(9999.990000, np.nan)
Wind['BZ, GSE, nT'] = Wind['BZ, GSE, nT'].replace(9999.990000, np.nan)
Wind['Vector Mag.,nT'] = Wind['Vector Mag.,nT'].replace(9999.990000, np.nan)
Wind['Field Magnitude,nT'] = Wind['Field Magnitude,nT'].replace(9999.990000, np.nan)
Wind['KP_Vx,km/s'] = Wind['KP_Vx,km/s'].replace(99999.900000, np.nan)
Wind['Kp_Vy, km/s'] = Wind['Kp_Vy, km/s'].replace(99999.900000, np.nan)
Wind['KP_Vz, km/s'] = Wind['KP_Vz, km/s'].replace(99999.900000, np.nan)
Wind['KP_Speed, km/s'] = Wind['KP_Speed, km/s'].replace(99999.900000, np.nan)
Wind['Kp_proton Density, n/cc'] = Wind['Kp_proton Density, n/cc'].replace(999.990000, np.nan)
Wind['Wind, Xgse,Re'] = Wind['Wind, Xgse,Re'].replace(9999.990000, np.nan)
Wind['Wind, Ygse,Re'] = Wind['Wind, Ygse,Re'].replace(9999.990000, np.nan)
Wind['Wind, Zgse,Re'] = Wind['Wind, Zgse,Re'].replace(9999.990000, np.nan)

Wind['Wind, Xgse,Re'] = Wind['Wind, Xgse,Re']*Re
Wind['Wind, Ygse,Re'] = Wind['Wind, Ygse,Re']*Re
Wind['Wind, Zgse,Re'] = Wind['Wind, Zgse,Re']*Re

#%%

wnd_nan_inds = []

columns_to_check = ['Kp_proton Density, n/cc', 'BZ, GSE, nT', 'KP_Vx,km/s']

for column in columns_to_check:
    nan_indices = Wind[Wind[column].isna()].index
    wnd_nan_inds.append(nan_indices)
    
    successive_nan_lengths = []
    current_length = 0
    
    for i in range(1, len(nan_indices)):
        if nan_indices[i] == nan_indices[i - 1] + 1:
            current_length += 1
        else:
            if current_length > 0:
                successive_nan_lengths.append(current_length + 1)
                current_length = 0

    if current_length > 0:
        successive_nan_lengths.append(current_length + 1)

    #wnd_nan_inds.append(successive_nan_lengths)
    
    
#%%

wnd_n_nans, wnd_Bz_nans, wnd_v_nans = wnd_nan_inds


#%%

ACE = pd.read_csv('ACE_data_unix.csv')

#%%

ace_nan_inds = []

columns_to_check = ['n', 'Bz', 'vx']

for column in columns_to_check:
    nan_indices = ACE[ACE[column].isna()].index
    ace_nan_inds.append(nan_indices)
    
    successive_nan_lengths = []
    current_length = 0
    
    for i in range(1, len(nan_indices)):
        if nan_indices[i] == nan_indices[i - 1] + 1:
            current_length += 1
        else:
            if current_length > 0:
                successive_nan_lengths.append(current_length + 1)
                current_length = 0

    if current_length > 0:
        successive_nan_lengths.append(current_length + 1)

    #ace_nan_inds.append(successive_nan_lengths)
    
    
#%%

ace_n_nans, ace_Bz_nans, ace_v_nans = ace_nan_inds

#%%

# Assuming you have three datasets named df1, df2, and df3
# Replace dataset names accordingly
datasets = [ACE, DSCOVR, Wind]

max_allowed_nans = 10  # You can change this value as needed

def filter_good_data(df, column_index):
    nan_mask = df.iloc[:, column_index].isnull()
    consecutive_nans = nan_mask.groupby((~nan_mask).cumsum()).cumsum()
    good_regions = consecutive_nans <= max_allowed_nans
    start_indices = good_regions.index[good_regions & ~good_regions.shift(fill_value=False)].tolist()
    end_indices = good_regions.index[~good_regions & good_regions.shift(fill_value=False)].tolist()
    
    gaps = [start_indices[i + 1] - end_indices[i] for i in range(len(start_indices) - 1)]
    gaps_in_hours = np.asarray(gaps) / 60
    
    good_data = np.array([[df['Time'][start], df['Time'][end]] for start, end in zip(start_indices, end_indices)])
    
    return good_data

all_good_data = []

# Assuming you have the same number of columns in each dataset
num_columns = len(datasets[0].columns)

for dataset in datasets:
    dataset_good_data = []
    for col_index in range(num_columns):
        col_good_data = filter_good_data(dataset, col_index)
        dataset_good_data.append(col_good_data)
    all_good_data.append(dataset_good_data)

# Check if all datasets have the same number of good data regions for each column
num_regions_per_column = [[len(good_data) for good_data in dataset_good_data] for dataset_good_data in all_good_data]

if all(all(num == num_regions_per_column[0][0] for num in row) for row in num_regions_per_column):
    print("All datasets have the same number of good data regions for each column.")
else:
    print("Warning: Number of good data regions is different across datasets.")

# Now 'all_good_data' contains a list for each dataset with good data intervals for each column
# You can use these intervals for further processing or filtering
