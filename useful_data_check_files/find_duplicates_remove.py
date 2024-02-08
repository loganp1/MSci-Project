# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 19:46:18 2024

@author: logan
"""

import sys
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

# Create dictionary of dataframes
sc_dict = {'ACE': df_ACE, 'DSCOVR': df_DSCOVR, 'Wind': df_Wind}

# Load this into Space_Weather_Forecast and watch the magic work!
myclass = Space_Weather_Forecast(SC_dict=sc_dict, SYM_real=df_SYM)


#%% Find discepancy between length of SYM vs SC data

# Compare the lengths of the DataFrames
if len(df_SYM) != len(df_ACE):
    print(f"Length mismatch: df_SYM has {len(df_SYM)} rows, df_ACE has {len(df_ACE)} rows.")

    # Find discrepancies in the 'DateTime' column
    sym_datetime_set = set(df_SYM['DateTime'])
    ace_datetime_set = set(df_ACE['DateTime'])

    sym_unique_datetimes = sym_datetime_set - ace_datetime_set
    ace_unique_datetimes = ace_datetime_set - sym_datetime_set

    print(f"Unique datetimes in df_SYM but not in df_ACE: {sym_unique_datetimes}")
    print(f"Unique datetimes in df_ACE but not in df_SYM: {ace_unique_datetimes}")
else:
    print("The lengths of df_SYM and df_ACE are the same.")


#%% Find duplicates

# Check for duplicates in 'DateTime' column
duplicates_SYM = df_SYM[df_SYM.duplicated('DateTime', keep=False)]
duplicates_ACE = df_ACE[df_ACE.duplicated('DateTime', keep=False)]

print(f"Duplicates in df_SYM:\n{duplicates_SYM}")
print(f"Duplicates in df_ACE:\n{duplicates_ACE}")

#%% Drop duplicates from df_SYM

# Identify and drop duplicates in df_SYM
df_SYM = df_SYM.drop_duplicates(subset='DateTime', keep='first')

# Check if duplicates are removed
duplicates_SYM = df_SYM[df_SYM.duplicated('DateTime', keep=False)]
print(f"Duplicates in df_SYM after removal:\n{duplicates_SYM}")

# Save the new DataFrame to a CSV file
df_SYM.to_csv('SYM_data_unix.csv', index=False)
