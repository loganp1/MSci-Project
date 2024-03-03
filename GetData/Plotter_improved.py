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
multiT=np.load('MvsR_deltaTs_improved.npy')

multi_NI=np.load('MvsR_maxCCs_NOtimeinterp.npy')
multiT_NI=np.load('MvsR_deltaTs_NOtimeinterp.npy')

# CCs for ACE vs real
ACEvals=np.load('AvsR_maxCCs_improved.npy')
ACET=np.load('AvsR_deltaTs_improved.npy')

# CCs for DSCOVR vs real
DSCvals=np.load('DvsR_maxCCs_improved.npy')
DSCT=np.load('DvsR_deltaTs_improved.npy')

# CCs for Wind vs real
Windvals=np.load('WvsR_maxCCs_improved.npy')
WindT=np.load('WvsR_deltaTs_improved.npy')

# CCs for OMNI predictions vs real
OMNIvals=np.load('OvsR_maxCCs.npy')
OMNIT=np.load('OvsR_deltaTs.npy')

# The pairs of spacecraft that we will compare to multi
#zvCCsAD = np.load('ADvsR_zvCCs_improved.npy')
ADvals = np.load('ADvsR_maxCCs_improved.npy')
ADT = np.load('ADvsR_deltaTs_improved.npy')

#zvCCsAW = np.load('AWvsR_zvCCs_improved.npy')
AWvals = np.load('AWvsR_maxCCs_improved.npy')
AWT = np.load('AWvsR_deltaTs_improved.npy')

#zvCCsDW = np.load('DWvsR_zvCCs_improved.npy')
DWvals = np.load('DWvsR_maxCCs_improved.npy')
DWT = np.load('DWvsR_deltaTs_improved.npy')

# NO time interpolated
#zvCCsAD = np.load('ADvsR_zvCCs_NOtimeinterp.npy')
ADvals_NI = np.load('ADvsR_maxCCs_NOtimeinterp.npy')
ADT_NI = np.load('ADvsR_deltaTs_NOtimeinterp.npy')

#zvCCsAW = np.load('AWvsR_zvCCs_NOtimeinterp.npy')
AWvals_NI = np.load('AWvsR_maxCCs_NOtimeinterp.npy')
AWT_NI = np.load('AWvsR_deltaTs_NOtimeinterp.npy')

#zvCCsDW = np.load('DWvsR_zvCCs_NOtimeinterp.npy')
DWvals_NI = np.load('DWvsR_maxCCs_NOtimeinterp.npy')
DWT_NI = np.load('DWvsR_deltaTs_NOtimeinterp.npy')

#%%


filter_time = 2400

maskm = abs(multiT)<filter_time
maskA = abs(ACET)<filter_time
maskD = abs(DSCT)<filter_time
maskW = abs(WindT)<filter_time

f_multi = multi[maskm]
f_ACE = ACEvals[maskm]
f_DSC = DSCvals[maskm]
f_Wind = Windvals[maskm]
f_OMNI = OMNIvals[maskm]

#Meanvals = (ACEvals+DSCvals+Windvals)/3
Meanvals = (f_ACE+f_DSC+f_Wind)/3

#plt.figure(figsize=(10,7))

# with plt.style.context('ggplot'):
#     sns.set_palette("tab10")
#     plt.hist([multi,ACEvals,DSCvals,Windvals,OMNIvals],density=True)
    
with plt.style.context('ggplot'):
    sns.set_palette("tab10")
    plt.hist([f_multi,f_ACE,f_DSC,f_Wind,f_OMNI],density=True)

# Set fontsize for axes labels
plt.xlabel("Cross Correlation in SYM/H", fontsize=15,color='black')
plt.ylabel("Probability Density", fontsize=15,color='black')
plt.ylim(0, 3)

# Uncomment the line below if you want to set a title with fontsize
# plt.title("Performance Comparison", fontsize=15)

# Set fontsize for tick labels on both axes
plt.xticks(fontsize=15,color='black')
plt.yticks(fontsize=15,color='black')

# Set fontsize for legend
plt.legend(["Multi", "ACE", "DSCOVR", "Wind", "OMNI"], fontsize=12, loc='upper left')

# Print mean values
print(np.mean(f_multi))
print(np.mean(f_ACE))
print(np.mean(f_DSC))
print(np.mean(f_Wind))
print(np.mean(f_OMNI))

# Show the plot
plt.show()

with plt.style.context('ggplot'):
    sns.set_palette("tab10")
    freqs, bins, _ = plt.hist([Meanvals,f_multi,f_OMNI],density=True)

# Set fontsize for axes labels
plt.xlabel("Cross Correlation in SYM/H", fontsize=15,color='black')
plt.ylabel("Probability Density", fontsize=15,color='black')
plt.ylim(0, 3)

# Uncomment the line below if you want to set a title with fontsize
# plt.title("Performance Comparison", fontsize=15)

# Set fontsize for tick labels on both axes
plt.xticks(fontsize=15,color='black')
plt.yticks(fontsize=15,color='black')

# Set fontsize for legend
plt.legend(["Mean Single", "Weighted Average",  "OMNI"], fontsize=12, loc='upper left')
print(np.mean(Meanvals))


#%% Look at the pairs of SC

# For time INTERPOLATED

filter_time = 2400

maskm = abs(multiT)<filter_time

f_multi = multi[maskm]
f_AD = ADvals[maskm]
f_AW = AWvals[maskm]
f_DW = DWvals[maskm]

    
with plt.style.context('ggplot'):
    sns.set_palette("tab10")
    plt.hist([f_multi,f_AD,f_AW,f_DW],density=True)

# Set fontsize for axes labels
plt.xlabel("Cross Correlation in SYM/H", fontsize=15,color='black')
plt.ylabel("Probability Density", fontsize=15,color='black')
plt.ylim(0, 3)

# Uncomment the line below if you want to set a title with fontsize
# plt.title("Performance Comparison", fontsize=15)

# Set fontsize for tick labels on both axes
plt.xticks(fontsize=15,color='black')
plt.yticks(fontsize=15,color='black')

# Set fontsize for legend
plt.legend(["Multi", "ACE-DSC", "ACE-Wind", "DSC-Wind"], fontsize=12, loc='upper left')


print(np.mean(f_multi))
print(np.mean(f_AD))
print(np.mean(f_AW))
print(np.mean(f_DW))

#%% For time NOT interpolated

filter_time = 2400

maskm = abs(multiT)<filter_time

f_multi_NI = multi_NI[maskm]
f_AD_NI = ADvals_NI[maskm]
f_AW_NI = AWvals_NI[maskm]
f_DW_NI = DWvals_NI[maskm]

    
with plt.style.context('ggplot'):
    sns.set_palette("tab10")
    plt.hist([f_multi,f_AD,f_AW,f_DW],density=True)

# Set fontsize for axes labels
plt.xlabel("Cross Correlation in SYM/H", fontsize=15,color='black')
plt.ylabel("Probability Density", fontsize=15,color='black')
plt.ylim(0, 3)

# Uncomment the line below if you want to set a title with fontsize
# plt.title("Performance Comparison", fontsize=15)

# Set fontsize for tick labels on both axes
plt.xticks(fontsize=15,color='black')
plt.yticks(fontsize=15,color='black')

# Set fontsize for legend
plt.legend(["Multi", "ACE-DSC", "ACE-Wind", "DSC-Wind"], fontsize=12, loc='upper left')


print(np.mean(f_multi_NI))
print(np.mean(f_AD_NI))
print(np.mean(f_AW_NI))
print(np.mean(f_DW_NI))

#%% Proportions of most accurate predictions (highest CC in each period)

# Time INTERPOLATED

# Initialize a list to store the indices of the lists with the highest values
max_indices = []

# Loop through the indices
for i in range(len(f_multi)):
    values_at_index = [f_multi[i], f_AD[i], f_AW[i], f_DW[i]]
    max_index = values_at_index.index(max(values_at_index))
    max_indices.append(max_index)

# Count occurrences of each index to create proportions for the pie chart
counts = [max_indices.count(0), max_indices.count(1), max_indices.count(2), max_indices.count(3)]

# Labels for the pie chart
labels = ['ALL 3', 'ACE-DSCOVR', 'ACE-Wind', 'DSCOVR-Wind']

# Create pie chart
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Most Accurate Predictions - Time Interpolated')
plt.show()

#%% Proportions of most accurate predictions (highest CC in each period)

# NOT time interp

# Initialize a list to store the indices of the lists with the highest values
max_indices = []

# Loop through the indices
for i in range(len(f_multi)):
    values_at_index = [f_multi_NI[i], f_AD_NI[i], f_AW_NI[i], f_DW_NI[i]]
    max_index = values_at_index.index(max(values_at_index))
    max_indices.append(max_index)

# Count occurrences of each index to create proportions for the pie chart
counts = [max_indices.count(0), max_indices.count(1), max_indices.count(2), max_indices.count(3)]

# Labels for the pie chart
labels = ['ALL 3', 'ACE-DSCOVR', 'ACE-Wind', 'DSCOVR-Wind']

# Create pie chart
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Most Accurate Predictions - NOT Time Interpolated')
plt.show()

#%% For mean 2 vs all 3

# Time INTERPOLATED

mean_2SC = (f_AD+f_AW+f_DW)/3

# Initialize a list to store the indices of the lists with the highest values
max_indices = []

# Loop through the indices
for i in range(len(f_multi)):
    values_at_index = [f_multi[i], mean_2SC[i]]
    max_index = values_at_index.index(max(values_at_index))
    max_indices.append(max_index)

# Count occurrences of each index to create proportions for the pie chart
counts = [max_indices.count(0), max_indices.count(1)]

# Labels for the pie chart
labels = ['ALL 3', 'Mean Pairs']

# Create pie chart
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Lists with Highest Values')
plt.show()

#%% For mean 2 vs all 3

# NO interpolation

mean_2SC_NI = (f_AD_NI+f_AW_NI+f_DW_NI)/3

# Initialize a list to store the indices of the lists with the highest values
max_indices = []

# Loop through the indices
for i in range(len(f_multi_NI)):
    values_at_index = [f_multi_NI[i], mean_2SC_NI[i]]
    max_index = values_at_index.index(max(values_at_index))
    max_indices.append(max_index)

# Count occurrences of each index to create proportions for the pie chart
counts = [max_indices.count(0), max_indices.count(1)]

# Labels for the pie chart
labels = ['ALL 3', 'Mean Pairs']

# Create pie chart
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Lists with Highest Values')
plt.show()


#%% Multi vs AD

# Time INTERPOLATED

# Initialize a list to store the indices of the lists with the highest values
max_indices = []

# Loop through the indices
for i in range(len(f_multi)):
    values_at_index = [f_multi[i], f_AD[i]]
    max_index = values_at_index.index(max(values_at_index))
    max_indices.append(max_index)

# Count occurrences of each index to create proportions for the pie chart
counts = [max_indices.count(0), max_indices.count(1)]

# Labels for the pie chart
labels = ['ALL 3', 'AD']

# Create pie chart
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Lists with Highest Values')
plt.show()

#%% Multi vs AD

# NO interp

# Initialize a list to store the indices of the lists with the highest values
max_indices = []

# Loop through the indices
for i in range(len(f_multi_NI)):
    values_at_index = [f_multi_NI[i], f_AD_NI[i]]
    max_index = values_at_index.index(max(values_at_index))
    max_indices.append(max_index)

# Count occurrences of each index to create proportions for the pie chart
counts = [max_indices.count(0), max_indices.count(1)]

# Labels for the pie chart
labels = ['ALL 3', 'AD']

# Create pie chart
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Lists with Highest Values')
plt.show()

#%% Multi vs AW

# Time INTERPOLATED

# Initialize a list to store the indices of the lists with the highest values
max_indices = []

# Loop through the indices
for i in range(len(f_multi)):
    values_at_index = [f_multi[i], f_AW[i]]
    max_index = values_at_index.index(max(values_at_index))
    max_indices.append(max_index)

# Count occurrences of each index to create proportions for the pie chart
counts = [max_indices.count(0), max_indices.count(1)]

# Labels for the pie chart
labels = ['ALL 3', 'AW']

# Create pie chart
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Lists with Highest Values')
plt.show()

#%% Multi vs AW

# NO interp

# Initialize a list to store the indices of the lists with the highest values
max_indices = []

# Loop through the indices
for i in range(len(f_multi_NI)):
    values_at_index = [f_multi_NI[i], f_AW_NI[i]]
    max_index = values_at_index.index(max(values_at_index))
    max_indices.append(max_index)

# Count occurrences of each index to create proportions for the pie chart
counts = [max_indices.count(0), max_indices.count(1)]

# Labels for the pie chart
labels = ['ALL 3', 'AW']

# Create pie chart
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Lists with Highest Values')
plt.show()

#%% Multi vs DW

# Time INTERP

# Initialize a list to store the indices of the lists with the highest values
max_indices = []

# Loop through the indices
for i in range(len(f_multi)):
    values_at_index = [f_multi[i], f_DW[i]]
    max_index = values_at_index.index(max(values_at_index))
    max_indices.append(max_index)

# Count occurrences of each index to create proportions for the pie chart
counts = [max_indices.count(0), max_indices.count(1)]

# Labels for the pie chart
labels = ['ALL 3', 'DW']

# Create pie chart
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Lists with Highest Values')
plt.show()

#%% Multi vs DW

# NO interp

# Initialize a list to store the indices of the lists with the highest values
max_indices = []

# Loop through the indices
for i in range(len(f_multi_NI)):
    values_at_index = [f_multi_NI[i], f_DW_NI[i]]
    max_index = values_at_index.index(max(values_at_index))
    max_indices.append(max_index)

# Count occurrences of each index to create proportions for the pie chart
counts = [max_indices.count(0), max_indices.count(1)]

# Labels for the pie chart
labels = ['ALL 3', 'DW']

# Create pie chart
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Proportions of Lists with Highest Values')
plt.show()