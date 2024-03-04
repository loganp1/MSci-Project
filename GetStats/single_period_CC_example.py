# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:08:07 2024

@author: logan
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

time_delays = np.load('time_delays_example.npy')
crossCs = np.load('cross_corr_example.npy')

with plt.style.context('ggplot'):
    plt.plot(time_delays,crossCs)
    
plt.ylabel('Cross Correlation')
plt.xlabel('Time Shift (s)')
