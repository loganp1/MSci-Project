# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:16:17 2024

@author: logan
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add path to the directory containing classes
sys.path.append(r"C:\Users\logan\OneDrive - Imperial College London\Uni\Year 4\MSci Project\MSci-Project"
                r"\Final_Model_OOP")

# Now you can import modules from these directories
from SC_Propagation_CLASS import SC_Propagation
from SYM_H_Model_CLASS import SYM_H_Model
#from Space_Weather_Forecasting_CLASS import Space_Weather_Forecast

#%%

# Load data
df_ACE = pd.read_csv('ace_T2_1min_unix.csv')
df_DSCOVR = pd.read_csv('dscovr_T2_1min_unix.csv')
df_Wind = pd.read_csv('wind_T2_1min_unix.csv')
df_SYM = pd.read_csv('SYM2_unix.csv')

# Extract initial SYM/H value ready for forecasting
sym0 = df_SYM['SYM/H, nT'].values[0]

# Create dictionary of dataframes
sc_dict = {'ACE': df_ACE, 'DSCOVR': df_DSCOVR, 'Wind': df_Wind}

#%%

# Create an object from class to propagate the data downstream
class1 = SC_Propagation(sc_dict)

# Put the data in this class in required form for propagation (probably should do this auto in class)
class1.required_form()


#%% Propagate each spacecraft using single spacecraft method

df_prop_ace = class1.singleSC_Propagate('ACE')
df_prop_dsc = class1.singleSC_Propagate('DSCOVR')
df_prop_wind = class1.singleSC_Propagate('Wind')

#%% Propagate combined using WA method

df_prop_multi = class1.multiSC_WA_Propagate()

#%% Now test 2nd class - SYM/H forecasting

# Test single spacecraft - just do ACE for now
class2a = SYM_H_Model(df_prop_ace,sym0)
sym_forecast_single = class2a.predict_SYM()

# Multi spacecraft
class2b = SYM_H_Model(df_prop_multi,sym0)
sym_forecast_multi = class2b.predict_SYM()

#%% Plot SYM/H forecast

# Single
plt.plot(df_prop_ace['Time'],sym_forecast_single)
plt.show()

# Multi
plt.plot(df_prop_multi['Time'],sym_forecast_multi)
plt.show()
