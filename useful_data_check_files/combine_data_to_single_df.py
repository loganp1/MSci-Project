# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 19:01:39 2024

@author: logan
"""

import pandas as pd

df1 = pd.read_csv('ace_T2_1min_unix.csv')
df2 = pd.read_csv('ace_T3_1min_unix.csv')
df3 = pd.read_csv('ace_T4_1min_unix.csv')

# Concatenate the DataFrames vertically
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Convert the 'Time' column to datetime objects
combined_df['DateTime'] = pd.to_datetime(combined_df['Time'], unit='s')

# Sort the combined DataFrame by the 'DateTime' column
combined_df.sort_values(by='DateTime', inplace=True)

# Save the combined DataFrame to a CSV file
combined_df.to_csv('ace_data_unix.csv', index=False)


#%%

df1 = pd.read_csv('dscovr_T2_1min_unix.csv')
df2 = pd.read_csv('dscovr_T3_1min_unix.csv')
df3 = pd.read_csv('dscovr_T4_1min_unix.csv')

# Concatenate the DataFrames vertically
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Convert the 'Time' column to datetime objects
combined_df['DateTime'] = pd.to_datetime(combined_df['Time'], unit='s')

# Sort the combined DataFrame by the 'DateTime' column
combined_df.sort_values(by='DateTime', inplace=True)

# Save the combined DataFrame to a CSV file
combined_df.to_csv('dscovr_data_unix.csv', index=False)


#%%

df1 = pd.read_csv('wind_T2_1min_unix.csv')
df2 = pd.read_csv('wind_T3_1min_unix.csv')
df3 = pd.read_csv('wind_T4_1min_unix.csv')

# Concatenate the DataFrames vertically
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Convert the 'Time' column to datetime objects
combined_df['DateTime'] = pd.to_datetime(combined_df['Time'], unit='s')

# Sort the combined DataFrame by the 'DateTime' column
combined_df.sort_values(by='DateTime', inplace=True)

# Save the combined DataFrame to a CSV file
combined_df.to_csv('wind_data_unix.csv', index=False)


#%%

# Read individual DataFrames from CSV files
df1 = pd.read_csv('SYM1_unix.csv')
df2 = pd.read_csv('SYM2_unix.csv')
df3 = pd.read_csv('SYM3_unix.csv')
df4 = pd.read_csv('SYM4_unix.csv')

# Concatenate the DataFrames vertically
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Convert the 'Time' column to datetime objects
combined_df['DateTime'] = pd.to_datetime(combined_df['Time'], unit='s')

# Filter the DataFrame for the specified date range
start_date = pd.to_datetime('2016-06-28 00:00:00')
end_date = pd.to_datetime('2019-06-27 23:59:59')

filtered_df = combined_df[(combined_df['DateTime'] >= start_date) & (combined_df['DateTime'] <= end_date)]

# Sort the filtered DataFrame by the 'DateTime' column
filtered_df.sort_values(by='DateTime', inplace=True)

# Save the filtered and sorted DataFrame to a CSV file
filtered_df.to_csv('SYM_data_unix.csv', index=False)
