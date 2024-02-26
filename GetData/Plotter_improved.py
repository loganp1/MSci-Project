# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 19:24:02 2024

@author: logan
"""

# Import modules

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import seaborn as sns

# CCs for multi vs real
multi=np.load('MvsR_maxCCs_improved.npy')

# CCs for ACE vs real
ACEvals=np.load('AvsR_maxCCs_improved.npy')

# CCs for DSCOVR vs real
DSCvals=np.load('DvsR_maxCCs_improved.npy')

# CCs for Wind vs real
Windvals=np.load('WvsR_maxCCs_improved.npy')

# CCs for OMNI predictions vs real
OMNIvals=np.load('OvsR_maxCCs.npy')

#plt.figure(figsize=(10,7))

with plt.style.context('ggplot'):
    sns.set_palette("tab10")
    plt.hist([multi,ACEvals,DSCvals,Windvals,OMNIvals],density=True)

# Set fontsize for axes labels
plt.xlabel("Cross Correlation in SYM/H", fontsize=15)
plt.ylabel("Probability Density", fontsize=15)
plt.ylim(0, 2.5)

# Uncomment the line below if you want to set a title with fontsize
# plt.title("Performance Comparison", fontsize=15)

# Set fontsize for tick labels on both axes
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Set fontsize for legend
plt.legend(["Multi", "ACE", "DSCOVR", "Wind", "OMNI"], fontsize=12, loc='upper left')

# Print mean values
print(np.mean(multi))
print(np.mean(ACEvals))
print(np.mean(DSCvals))
print(np.mean(Windvals))
print(np.mean(OMNIvals))

# Show the plot
plt.show()