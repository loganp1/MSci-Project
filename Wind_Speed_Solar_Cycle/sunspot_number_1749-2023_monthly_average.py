# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:19:22 2023

@author: logan
"""

import pandas as pd
import matplotlib.pyplot as plt

# Process the data

df_sunspots = pd.read_csv('sunspots_1749-2023_monthly_average.csv')#,delimiter=';',header=None)

#col_names = ['Year','Month','year_val','mean','std_dev','no_obs','valid_check']

#df_sunspots.columns = col_names

#df_sunspots.to_csv('sunspots_1749-2023_monthly_average.csv')

# Now saved the csv with new columns so shouldn't need this again

#%%

# Create shortened timescale dataframe

df_sunspots = df_sunspots[df_sunspots['Year'] >= 1995]

df_sunspots.to_csv('sunspots_1995-2023_monthly_average.csv')

#%%

# Plot sunspot number using the numerical year value in column 3, 
# which gives month as fraction of year from centre of month

year = df_sunspots['year_val']
spot_num = df_sunspots['mean']

plt.plot(year,spot_num)

