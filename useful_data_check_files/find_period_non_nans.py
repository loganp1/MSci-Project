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

Q = 1 # for this code i.e. it finds longest run without any nans

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