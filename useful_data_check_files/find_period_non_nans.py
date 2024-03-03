# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:24:59 2024

@author: logan
"""

import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add path to the directory containing classes
sys.path.append(r"C:\Users\logan\OneDrive - Imperial College London\Uni\Year 4\MSci Project\MSci-Project"
                r"\Final_Model_OOP")

from Space_Weather_Forecasting_CLASS import Space_Weather_Forecast


#%%

# Load data
df_ACE = pd.read_csv('ace_data_unix.csv')
df_DSCOVR = pd.read_csv('dscovr_data_unix.csv')
df_Wind = pd.read_csv('wind_data_unix.csv')
df_SYM = pd.read_csv('SYM_data_unix.csv')

#%%

# Create dictionary of dataframes
sc_dict = {'ACE': df_ACE, 'DSCOVR': df_DSCOVR, 'Wind': df_Wind}

# Load this into Space_Weather_Forecast and watch the magic work!
myclass = Space_Weather_Forecast(SC_dict=sc_dict, SYM_real=df_SYM)


#%% Find longest run of non-nan data in n (density) column of ACE

# Convert the 'DateTime' column to datetime objects
df_ACE['DateTime'] = pd.to_datetime(df_ACE['DateTime'])

# Find consecutive runs of non-NaN values in 'n' column
df_ACE['n_not_nan'] = df_ACE['n'].notna().astype(int)
df_ACE['group'] = (df_ACE['n_not_nan'].ne(df_ACE['n_not_nan'].shift()) & df_ACE['n_not_nan']).cumsum()

# Calculate the length of each run
run_lengths = df_ACE.groupby('group')['n_not_nan'].transform('count')

# Find the index of the longest run
idx_longest_run = run_lengths.idxmax()

# Get the start and end dates of the longest run
longest_run_start = df_ACE.loc[idx_longest_run, 'DateTime']
longest_run_end = df_ACE.loc[idx_longest_run + run_lengths.max() - 1, 'DateTime']

print(f"The longest run without NaNs in 'n' column starts at {longest_run_start} and ends at {longest_run_end}.")

# Print additional information for debugging
print(f"Length of the entire DataFrame: {len(df_ACE)}")
print(f"Maximum run length: {run_lengths.max()}")


#%% Find longest run without Q nans by forming list of nan indices

Q = 9 # for this code i.e. it finds longest run without any nans

nanlist = df_ACE.index[pd.isnull(df_ACE['n'])].tolist()

# Extract the relevant values from df_ACE
x_values = df_ACE['Time'].iloc[nanlist].values
y_values = np.ones_like(x_values)  # Creating an array of 1's with the same shape as x_values
#%%
# Plot the scatter plot
plt.scatter(x_values, y_values,s=0.01)

#plt.xlim(min(df_ACE['Time']),max(df_ACE['Time']))

# Show the plot
plt.show()


#%% Find longest run without Q nans by forming list of nan indices

Q = 1 # for this code i.e. it finds longest run without any nans

def max_difference_values(lst):
    if len(lst) < 2:
        return None, None  # Not enough elements to calculate a difference

    max_difference = float('-inf')
    max_values = None, None

    for i in range(len(lst) - 1):
        difference = abs(lst[i + 1] - lst[i])
        if difference > max_difference:
            max_difference = difference
            max_values = lst[i], lst[i + 1]

    return max_values

values = max_difference_values(nanlist)
print(f"Values with the largest difference: {values}")


#%% Let's look at Bz and v as well

nan_indices = np.where(df_ACE['Bz'].isna())[0]

print("Indices of NaN values in df_ACE['Bz']:", nan_indices)

def longest_consecutive_run(indices):

    current_run = [indices[0]]
    longest_run = []

    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_run.append(indices[i])
        else:
            if len(current_run) > len(longest_run):
                longest_run = current_run
            current_run = [indices[i]]

    # Check for the last run
    if len(current_run) > len(longest_run):
        longest_run = current_run

    return longest_run

result = longest_consecutive_run(nan_indices)

print("Longest run of consecutive nans in Bz:", len(result), 'long')

#%%

nan_indices1 = np.where(df_ACE['vx'].isna())[0]

print("Indices of NaN values in df_ACE['Bz']:", nan_indices1)

def longest_consecutive_run(indices):

    current_run = [indices[0]]
    longest_run = []

    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_run.append(indices[i])
        else:
            if len(current_run) > len(longest_run):
                longest_run = current_run
            current_run = [indices[i]]

    # Check for the last run
    if len(current_run) > len(longest_run):
        longest_run = current_run

    return longest_run

result1 = longest_consecutive_run(nan_indices1)

print("Longest run of consecutive nans in v:", len(result1), 'long')

#%%

def consecutive_run_lengths(indices):

    run_lengths = []
    current_run_length = 1

    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_run_length += 1
        else:
            if current_run_length > 10:
                run_lengths.append(current_run_length)
            current_run_length = 1

    # Check for the last run
    if current_run_length > 10:
        run_lengths.append(current_run_length)

    return run_lengths

lengths_greater_than_10 = consecutive_run_lengths(nan_indices)

print("Lengths of consecutive sets greater than 10:", lengths_greater_than_10)

lengths_greater_than_10_v = consecutive_run_lengths(nan_indices1)

print("Lengths of consecutive sets greater than 10:", lengths_greater_than_10_v)
