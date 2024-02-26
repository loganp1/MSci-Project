# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:53:00 2024

@author: logan
"""

'''
Before I do all of this analysis we need to have an exact forecasting process with an offset so we can associate
an exact prediction at a specific time with the real value - so we can get the extreme values at the correct
points. 
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import seaborn as sns

# Load data
# multi_sym = np.load('multi_sym_forecast.npy')
# ace_sym = np.load('ace_sym_forecast.npy')
# dsc_sym = np.load('dscovr_sym_forecast.npy')
# wind_sym = np.load('wind_sym_forecast.npy')
# sym = pd.read_csv('SYM_data_unix.csv')

multi_sym = np.load('multi_sym_forecastnpy.npy')
ace_sym = np.load('ace_sym_forecastnpy.npy')
dsc_sym = np.load('dscovr_sym_forecastnpy.npy')
wind_sym = np.load('wind_sym_forecastnpy.npy')
sym = pd.read_csv('SYM_data_unix.csv')

times=np.load("split_data_times.npy")

# Big storm, good prediction:
x_range_start = 1.535e9
x_range_end = 1.536e9

# x_range_start = 1.526e9
# x_range_end = 1.527e9

# Function to filter data within the specified x-axis range
def filter_data(data, x_start, x_end):
    filtered_indices = [i for i, x in enumerate(data[0]) if x_start <= x <= x_end]
    filtered_data = [[data[0][i] for i in filtered_indices], [data[1][i] for i in filtered_indices]]
    return filtered_data

def filter_sym(data, x_start, x_end):
    return data[(data['Time'] >= x_start) & (data['Time'] <= x_end)]

# Filter data for sym DataFrame
filtered_sym = filter_sym(sym, x_range_start, x_range_end)

# Filter data for each variable
filtered_multi_sym = filter_data(multi_sym, x_range_start, x_range_end)
filtered_ace_sym = filter_data(ace_sym, x_range_start, x_range_end)
filtered_dsc_sym = filter_data(dsc_sym, x_range_start, x_range_end)
filtered_wind_sym = filter_data(wind_sym, x_range_start, x_range_end)

# Change to datetime variables
time0 = pd.to_datetime(filtered_multi_sym[0], unit='s')
time1 = pd.to_datetime(filtered_ace_sym[0],unit='s')
time2 = pd.to_datetime(filtered_dsc_sym[0],unit='s')
time3 = pd.to_datetime(filtered_wind_sym[0],unit='s')
time4 = pd.to_datetime(filtered_sym['DateTime'].values)
#%%
plt.style.use('ggplot')
sns.set_palette("tab10")
plt.figure(figsize=(10,5))
# Plot all four variables on the same graph
plt.plot(time0, filtered_multi_sym[1], label='Multi')
plt.plot(time1, filtered_ace_sym[1], label='ACE')
plt.plot(time2, filtered_dsc_sym[1], label='DSCOVR')
plt.plot(time3, filtered_wind_sym[1], label='Wind')
plt.plot(time4, filtered_sym['SYM/H, nT'].values, label='Real SYM/H')

# Set x-axis ticks to show only one label per day
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))

# Format x-axis tick labels as dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
fs=20
plt.xlabel("Date in 2018",fontsize=fs)
plt.ylabel("SYM/H [nT]",fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs)
plt.show()

#%%

x_range_start = 1.535e9
x_range_end = 1.536e9

testdf = filter_sym(sym, x_range_start, x_range_end)

time_test = testdf['Time'].values
sym_test = testdf['SYM/H, nT'].values

plt.plot(time_test,sym_test)


#%% 

# After a bit of visual analysis, our multi predictions seem to overshoot less than the single spacecrafts
# so I'll analyse this here, plot some sort of chart with the percentage a method is closest to extrema perhaps?

methods = ['ACE','DSCOVR','Wind','multi']

dataset=[ace_sym,dsc_sym,wind_sym,multi_sym]
closest_to_extrema = []
for i in range(len(times)): 
    period_minima = []
    for data in dataset:
        dataFilt = data[:, (data[0] >= times[i][0]) & (data[0] <= times[i][1])]
        period_minima.append(min(dataFilt[1]))
    max_index = period_minima.index(max(period_minima))
    # Add the corresponding method to closest_to_extrema
    closest_to_extrema.append(methods[max_index])
    print(i)
    
#%%

# Count the frequency of each element in the list
element_counts = {tuple(methods): closest_to_extrema.count(methods) for methods in methods}

# Explode a slice (optional, for emphasis)
explode = (0, 0, 0, 0)

colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']

# Create pie chart
plt.pie(element_counts.values(), autopct='%1.1f%%', startangle=140, 
         explode=explode, colors=colors, shadow=False, labeldistance=1.1,textprops={'fontsize': 10})

# Add legend
plt.legend(loc='upper right', labels=methods, bbox_to_anchor=(1.2, 1.1))

#plt.figtext(0.5, 0.01, '''Pie Chart Highlighting Which Method (Single or Multi-Spacecraft) Records
#Extreme SYM/H Values Nearest to the Extrema of the Real SYM/H Values''', ha='center', va='center')
plt.show()

# colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']
# plt.pie(element_counts.values(), autopct='%1.1f%%', startangle=140, 
#         explode=explode, colors=colors, shadow=False, labels=methods, labeldistance=1.1,textprops={'fontsize': 12})
# plt.show()


#%% Try a new chart type: Bar chart

import matplotlib.pyplot as plt

# Count the occurrences of each type
counts = {type_: closest_to_extrema.count(type_) for type_ in set(closest_to_extrema)}

# Plot a bar chart
plt.figure(figsize=(10, 6))
plt.bar(counts.keys(), counts.values(), color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Types')
plt.ylabel('Frequency')
plt.title('Frequency of Each Type')
plt.show()


#%% Extrema offset values

# After getting that single metric over the whole dataset I want something a bit more detailed that
# I can do some good analysis on
# So here I will take the 'offset' in extreme values, one for each spacecraft for each subDF
# I can then histogram those, hopefully seeing multi peak at the lower values relative to the singles

#%% Put real sym data into same format as forecasted syms

real_sym = np.array([sym['Time'].values,sym['SYM/H, nT'].values])

#%%

dataset=[ace_sym,dsc_sym,wind_sym,multi_sym]
extrema_offsets = []
extrema_indices = []
for i in range(len(times)): 
    realFilt = real_sym[:, (real_sym[0] >= times[i][0]) & (real_sym[0] <= times[i][1])]
    real_extrema = min(realFilt[1])
    index_real_extrema = real_sym[1].tolist().index(real_extrema)
    extrema_offsets_dataseti = []
    extrema_indices_dataseti = []
    for data in dataset:
        dataFilt = data[:, (data[0] >= times[i][0]) & (data[0] <= times[i][1])]
        extrema_offsets_dataseti.append(real_extrema - min(dataFilt[1]))
        index_extrema = dataFilt[1].tolist().index(min(dataFilt[1]))
        extrema_indices_dataseti.append(index_extrema)
    # Add the corresponding method to closest_to_extrema
    extrema_offsets.append(extrema_offsets_dataseti)
    extrema_indices.append(extrema_indices_dataseti)
    
    print(i)
    
#%%
# ChatGPT changes (vectorised?):
extrema_offsets = []
extrema_indices = []

for i in range(len(times)):
    realFilt = real_sym[:, (real_sym[0] >= times[i][0]) & (real_sym[0] <= times[i][1])]
    real_extrema = np.min(realFilt[1])
    index_real_extrema = np.argmin(realFilt[1])
    
    extrema_offsets_dataseti = []
    extrema_indices_dataseti = []

    for data in dataset:
        dataFilt = data[:, (data[0] >= times[i][0]) & (data[0] <= times[i][1])]
        extrema_offsets_dataseti.append(real_extrema - np.min(dataFilt[1]))
        index_extrema = np.argmin(dataFilt[1])
        extrema_indices_dataseti.append(index_extrema)

    extrema_offsets.append(extrema_offsets_dataseti)
    extrema_indices.append(extrema_indices_dataseti)

    print(i)
    
#%% Plot data as histogram

# Transpose the offset data
extr_offsets = np.asarray(extrema_offsets).T

# Extract each spacecraft's offsets
ace_offsets = abs(extr_offsets[0])
dsc_offsets = abs(extr_offsets[1])
wnd_offsets = abs(extr_offsets[2])
multi_offsets = abs(extr_offsets[3])
    
# Should hopefully see a higher frequency in smaller bins for multi
plt.hist([ace_offsets,dsc_offsets,wnd_offsets,multi_offsets],bins=100,label=['ACE','DSCOVR','Wind','Multi'],
         density=True)
plt.xlim(0,50)

plt.xlabel('Extrema Offset in SYM/H')
plt.ylabel('Probability Density')

plt.legend()
plt.show()

#%%

extreme_sym = []

for i in range(len(times)):
    
    realFilt = real_sym[:, (real_sym[0] >= times[i][0]) & (real_sym[0] <= times[i][1])]
    extreme_sym.append(min(realFilt[1]))
    print(i)

#%%

plt.hist(extreme_sym,bins=100)
