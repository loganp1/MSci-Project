# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:02:23 2024

@author: logan
"""

import matplotlib.pyplot as plt 
import numpy as np

#mCC_12hrs = np.load('MvsR_maxCCs_12hrs.npy')
mCC_4hrs = np.load('MvsR_maxCCs_4hrs.npy')
mCC_2hrs = np.load('MvsR_maxCCs_2hrs.npy')
mCC_8hrs = np.load('MvsR_maxCCs_8hrs.npy')
#mCC_10hrs = np.load('MvsR_maxCCs_10hrs.npy')

#%%

# Normalise the frequencies using density = True as each has different number of data points
with plt.style.context('ggplot'):
    plt.hist([mCC_4hrs,mCC_2hrs],density=True)
plt.legend(['4 hour periods','2 hour periods'])
plt.xlabel("Cross Correlation in SYM/H")
plt.ylabel("Probability Density")

print('4 hour mean:',np.mean(mCC_4hrs))
print('2 hour mean:',np.mean(mCC_2hrs))
