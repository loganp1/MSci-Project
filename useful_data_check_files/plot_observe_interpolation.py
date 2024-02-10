# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:27:55 2024

@author: logan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ace_data_unix.csv')

#%%

nan_indices = df[df['Bz'].isnull()].index.tolist()

print(len(nan_indices))

#%% Plot time vs n to see timescale of variations and what sort of range interpolation may be appr. for

plt.plot(df['Time'],df['Bz'])
plt.show()

# Interpolate linearly
df_interp = df.interpolate()

# Replot
plt.plot(df_interp['Time'],df_interp['Bz'])
plt.show()

#%% Velocity Magnitude

plt.plot(df['Time'],df['vx']**2+df['vy']**2+df['vz']**2)
plt.show()


#%% Plot v on top of n to compare if n misses any spikes

plt.plot(df['Time'],df['n'],label='n')
plt.plot(df['Time'],(df['vx']**2+df['vy']**2+df['vz']**2)/10000-8,label='v',alpha=0.5)

plt.legend()