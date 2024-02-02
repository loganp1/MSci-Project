# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 23:23:53 2024

@author: logan
"""

import pandas as pd

df = pd.read_csv('ace_T2_1min_unix.csv')

nan_count = df['vx'].isna().sum()
print(f"Number of NaN values in 'vx' column: {nan_count}")
print(f"Number of non-NaN values in 'vx' column: {len(df['vx'])-nan_count}")


df2 = pd.read_csv('ace_T3_1min_unix.csv')

nan_count2 = df2['vx'].isna().sum()
print(f"Number of NaN values in 'vx' column: {nan_count2}")
print(f"Number of non-NaN values in 'vx' column: {len(df['vx'])-nan_count2}")
