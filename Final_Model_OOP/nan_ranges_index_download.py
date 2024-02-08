# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:17:44 2024

@author: logan
"""

import pandas as pd
import numpy as np

df = pd.read_csv('ace_data_unix.csv')

nan_indices = df[df['n'].isnull()].index.tolist()

print(nan_indices)

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
