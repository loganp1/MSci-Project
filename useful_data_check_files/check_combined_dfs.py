# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 19:06:55 2024

@author: logan
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ace_data_unix.csv')

#%%

plt.plot(df['Time'],df['vx'])

#%%

is_ascending = df['Time'].is_monotonic_increasing
print(f"The 'Time' column is in ascending order: {is_ascending}")
