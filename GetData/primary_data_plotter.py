# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:47:33 2024

@author: logan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import matplotlib.dates as mdates
import pickle
import pandas as pd
from datetime import datetime
from cross_correlation import cross_correlation

#%%

# Load the list of DataFrames
with open('Filtered_SCdata_Bz_v_n_E_P_atL1.pkl', 'rb') as f:
    SCdata = pickle.load(f)

#%%

# Choose random period to compare density data between SC
period = 15

ACE_split_data = SCdata[0]
DSC_split_data = SCdata[1]
Wind_split_data = SCdata[2]

# 1st Period
tACE0 = ACE_split_data[period]['Time']
tDSC0 = DSC_split_data[period]['Time']
tWind0 = Wind_split_data[period]['Time']

nACE0 = ACE_split_data[period]['n']
nDSC0 = DSC_split_data[period]['Proton Density, n/cc']
nWind0 = Wind_split_data[period]['Kp_proton Density, n/cc']

tACE0_datetime = pd.to_datetime(tACE0, unit='s')
tDSC0_datetime = pd.to_datetime(tDSC0, unit='s')
tWind0_datetime = pd.to_datetime(tWind0, unit='s')

date = tACE0_datetime.iloc[0]
date_formatter = mdates.DateFormatter('%y-%m-%d')
date = mdates.date2num(date)
date = date_formatter(date)

plt.style.use('ggplot')
plt.plot(tACE0_datetime, nACE0, label='ACE')
plt.plot(tDSC0_datetime, nDSC0, label='DSCOVR')
plt.plot(tWind0_datetime, nWind0, label='Wind')
plt.legend()
plt.xlabel(date)
plt.ylabel('Solar Wind Plasma Density ($cm^{-3}$)')
plt.grid(True, which='major', linestyle='--', linewidth=1)
plt.tick_params(which='both')
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.show()


#%% Need some sort of quantitative comparison - look at range of densities in each period

ACE_n_ranges = []
DSC_n_ranges = []
Wind_n_ranges = []

ADc = [[],[]] 
AWc = [[],[]]
DWc = [[],[]]

for period in range(2730):
    
    ACE_min_n = min(SCdata[0][period]['n'])
    ACE_max_n = max(SCdata[0][period]['n'])
    ACE_n_ranges.append(ACE_max_n - ACE_min_n)
    
    DSC_min_n = min(SCdata[1][period]['Proton Density, n/cc'])
    DSC_max_n = max(SCdata[1][period]['Proton Density, n/cc'])
    DSC_n_ranges.append(DSC_max_n - DSC_min_n)
    
    Wind_min_n = min(SCdata[2][period]['Kp_proton Density, n/cc'])
    Wind_max_n = max(SCdata[2][period]['Kp_proton Density, n/cc'])
    Wind_n_ranges.append(Wind_max_n - Wind_min_n)
    
    ADc[1].append(cross_correlation(SCdata[0][period]['n'],SCdata[1][period]['Proton Density, n/cc'],
                  SCdata[0][period]['Time'].values)[1])
    ADc[0].append(cross_correlation(SCdata[0][period]['n'],SCdata[1][period]['Proton Density, n/cc'],
                  SCdata[0][period]['Time'].values)[0])
    
    AWc[1].append(cross_correlation(SCdata[0][period]['n'],SCdata[2][period]['Kp_proton Density, n/cc'],
                  SCdata[0][period]['Time'].values)[1])
    AWc[0].append(cross_correlation(SCdata[0][period]['n'],SCdata[2][period]['Kp_proton Density, n/cc'],
                  SCdata[0][period]['Time'].values)[0])
    
    DWc[1].append(cross_correlation(SCdata[1][period]['Proton Density, n/cc'],
                  SCdata[2][period]['Kp_proton Density, n/cc'], SCdata[0][period]['Time'].values)[1])
    DWc[0].append(cross_correlation(SCdata[1][period]['Proton Density, n/cc'],
                  SCdata[2][period]['Kp_proton Density, n/cc'], SCdata[0][period]['Time'].values)[0])
    
    print(period)
    
#%% Look at max-min ranges
    
freqs, bins, _ = plt.hist([ACE_n_ranges,DSC_n_ranges,Wind_n_ranges],bins=20,label=['ACE','DSCOVR','Wind'])
plt.legend()

#%% CCs

#plt.plot(ADc[0][1],ADc[1][1])
    
# Let's plot histograms of max CC for the pairs

AD_mCC = [max(CC) for CC in ADc[1]]
AW_mCC = [max(CC) for CC in AWc[1]]
DW_mCC = [max(CC) for CC in DWc[1]]

#%%

plt.hist([AD_mCC,AW_mCC,DW_mCC],label=['AD','AW','DW'],density=True)
plt.legend()
plt.xlabel('Cross Correlation')
plt.ylabel('Probability Density')
plt.title('Cross Correlations Between Density Data')

print(np.mean(AD_mCC))
print(np.mean(AW_mCC))
print(np.mean(DW_mCC))


#%% Try same again but with Bz data

ADc = [[],[]] 
AWc = [[],[]]
DWc = [[],[]]

for period in range(2730):
    
    # ACE_min_n = min(SCdata[0][period]['Bz'])
    # ACE_max_n = max(SCdata[0][period]['Bz'])
    # ACE_n_ranges.append(ACE_max_n - ACE_min_n)
    
    # DSC_min_n = min(SCdata[1][period]['Proton Density, n/cc'])
    # DSC_max_n = max(SCdata[1][period]['Proton Density, n/cc'])
    # DSC_n_ranges.append(DSC_max_n - DSC_min_n)
    
    # Wind_min_n = min(SCdata[2][period]['Kp_proton Density, n/cc'])
    # Wind_max_n = max(SCdata[2][period]['Kp_proton Density, n/cc'])
    # Wind_n_ranges.append(Wind_max_n - Wind_min_n)
    
    ADc[1].append(cross_correlation(SCdata[0][period]['Bz'],SCdata[1][period]['BZ, GSE, nT'],
                  SCdata[0][period]['Time'].values)[1])
    ADc[0].append(cross_correlation(SCdata[0][period]['Bz'],SCdata[1][period]['BZ, GSE, nT'],
                  SCdata[0][period]['Time'].values)[0])
    
    AWc[1].append(cross_correlation(SCdata[0][period]['Bz'],SCdata[2][period]['BZ, GSE, nT'],
                  SCdata[0][period]['Time'].values)[1])
    AWc[0].append(cross_correlation(SCdata[0][period]['Bz'],SCdata[2][period]['BZ, GSE, nT'],
                  SCdata[0][period]['Time'].values)[0])
    
    DWc[1].append(cross_correlation(SCdata[1][period]['BZ, GSE, nT'],
                  SCdata[2][period]['BZ, GSE, nT'], SCdata[0][period]['Time'].values)[1])
    DWc[0].append(cross_correlation(SCdata[1][period]['BZ, GSE, nT'],
                  SCdata[2][period]['BZ, GSE, nT'], SCdata[0][period]['Time'].values)[0])
    
    print(period)
    
#%% CCs

#plt.plot(ADc[0][1],ADc[1][1])
    
# Let's plot histograms of max CC for the pairs

AD_mCC = [max(CC) for CC in ADc[1]]
AW_mCC = [max(CC) for CC in AWc[1]]
DW_mCC = [max(CC) for CC in DWc[1]]

#%%

plt.hist([AD_mCC,AW_mCC,DW_mCC],label=['AD','AW','DW'],density=True)
plt.legend()
plt.xlabel('Cross Correlation')
plt.ylabel('Probability Density')
plt.title('Cross Correlations Between Density Data')

print(np.mean(AD_mCC))
print(np.mean(AW_mCC))
print(np.mean(DW_mCC))

#%%
# WOW ACE IS BAD FOR Bz!!! - do some plots

# Choose random period to compare density data between SC
period = 1

ACE_split_data = SCdata[0]
DSC_split_data = SCdata[1]
Wind_split_data = SCdata[2]

# 1st Period
tACE0 = ACE_split_data[period]['Time']
tDSC0 = DSC_split_data[period]['Time']
tWind0 = Wind_split_data[period]['Time']

nACE0 = ACE_split_data[period]['Bz']
nDSC0 = DSC_split_data[period]['BZ, GSE, nT']
nWind0 = Wind_split_data[period]['BZ, GSE, nT']

tACE0_datetime = pd.to_datetime(tACE0, unit='s')
tDSC0_datetime = pd.to_datetime(tDSC0, unit='s')
tWind0_datetime = pd.to_datetime(tWind0, unit='s')

date = tACE0_datetime.iloc[0]
date_formatter = mdates.DateFormatter('%y-%m-%d')
date = mdates.date2num(date)
date = date_formatter(date)

plt.style.use('ggplot')
plt.plot(tACE0_datetime, nACE0, label='ACE')
plt.plot(tDSC0_datetime, nDSC0, label='DSCOVR')
plt.plot(tWind0_datetime, nWind0, label='Wind')
plt.legend()
plt.xlabel(date)
plt.ylabel('Bz (nT)')
plt.grid(True, which='major', linestyle='--', linewidth=1)
plt.tick_params(which='both')
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.show()