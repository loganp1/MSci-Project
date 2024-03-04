# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 08:47:23 2024

@author: logan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 19:07:49 2024

@author: Ned
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from itertools import combinations
import seaborn as sns
from scipy.optimize import curve_fit

#%% Import data

# Our 4 hour period data point times
times=np.load("split_data_times.npy")

# Spacecraft data
ACE=pd.read_csv("ace_data_unix.csv")
Wind=pd.read_csv("wind_data_unix.csv")
DSCOVR=pd.read_csv("dscovr_data_unix.csv")

# Real SYM/H data
SYM=pd.read_csv("SYM_data_unix.csv")

# Cross correlations for the spacecraft pairs
CCs1 = np.load('AvsD_maxCCs.npy')
CCs0 = np.load('AvsW_maxCCs.npy')
CCs2 = np.load('DvsW_maxCCs.npy')

# CCs for multi vs real
multi=np.load('MvsR_maxCCs.npy')

# CCs for ACE vs real
ACEvals=np.load('AvsR_maxCCs.npy')

# CCs for OMNI predictions vs real
OMNIvals=np.load('OvsR_maxCCs.npy')

# CCs for DSCOVR vs real
dscvals=np.load('DvsR_maxCCs.npy')

# CCs for Wind vs real
windvals=np.load('WvsR_maxCCs.npy')

# Time shifts zero-val to max CC multi vs real
deltaT = np.load("MvsR_deltaTs.npy")

#%% Clean the spacecraft data

Re=6378
DSCOVR['Field Magnitude,nT'] = DSCOVR['Field Magnitude,nT'].replace(9999.99, np.nan)
DSCOVR['Vector Mag.,nT'] = DSCOVR['Vector Mag.,nT'].replace(9999.99, np.nan)
DSCOVR['BX, GSE, nT'] = DSCOVR['BX, GSE, nT'].replace(9999.99, np.nan)
DSCOVR['BY, GSE, nT'] = DSCOVR['BY, GSE, nT'].replace(9999.99, np.nan)
DSCOVR['BZ, GSE, nT'] = DSCOVR['BZ, GSE, nT'].replace(9999.99, np.nan)
DSCOVR['Speed, km/s'] = DSCOVR['Speed, km/s'].replace(99999.9, np.nan)
DSCOVR['Vx Velocity,km/s'] = DSCOVR['Vx Velocity,km/s'].replace(99999.9, np.nan)
DSCOVR['Vy Velocity, km/s'] = DSCOVR['Vy Velocity, km/s'].replace(99999.9, np.nan)
DSCOVR['Vz Velocity, km/s'] = DSCOVR['Vz Velocity, km/s'].replace(99999.9, np.nan)
DSCOVR['Proton Density, n/cc'] = DSCOVR['Proton Density, n/cc'].replace(999.999, np.nan)
DSCOVR['Wind, Xgse,Re'] = DSCOVR['Wind, Xgse,Re'].replace(9999.99, np.nan)
DSCOVR['Wind, Ygse,Re'] = DSCOVR['Wind, Ygse,Re'].replace(9999.99, np.nan)
DSCOVR['Wind, Zgse,Re'] = DSCOVR['Wind, Zgse,Re'].replace(9999.99, np.nan)

# Have to do change to km AFTER removing vals else fill value will change!
DSCOVR['Wind, Xgse,Re'] = DSCOVR['Wind, Xgse,Re']*Re
DSCOVR['Wind, Ygse,Re'] = DSCOVR['Wind, Ygse,Re']*Re
DSCOVR['Wind, Zgse,Re'] = DSCOVR['Wind, Zgse,Re']*Re

# Interpolate NaN values
for column in DSCOVR.columns:
    DSCOVR[column] = DSCOVR[column].interpolate()
    
# Define the desired order of columns as required for Ned's function: DSCOVR
desired_columns_order = [
    'Time',
    'Vx Velocity,km/s',
    'Vy Velocity, km/s',
    'Vz Velocity, km/s',
    'Wind, Xgse,Re',
    'Wind, Ygse,Re',
    'Wind, Zgse,Re',
    'BZ, GSE, nT',
    'Speed, km/s',
    'Proton Density, n/cc'
]

# Select only the desired columns and reorder them
DSCOVR = DSCOVR[desired_columns_order]

DSCOVR = DSCOVR.copy()

# Drop the original columns
DSCOVR = DSCOVR.drop(['Proton Density, n/cc', 'Speed, km/s', 'BZ, GSE, nT'], axis=1)




Wind['BX, GSE, nT'] = Wind['BX, GSE, nT'].replace(9999.990000, np.nan)
Wind['BY, GSE, nT'] = Wind['BY, GSE, nT'].replace(9999.990000, np.nan)
Wind['BZ, GSE, nT'] = Wind['BZ, GSE, nT'].replace(9999.990000, np.nan)
Wind['Vector Mag.,nT'] = Wind['Vector Mag.,nT'].replace(9999.990000, np.nan)
Wind['Field Magnitude,nT'] = Wind['Field Magnitude,nT'].replace(9999.990000, np.nan)
Wind['KP_Vx,km/s'] = Wind['KP_Vx,km/s'].replace(99999.900000, np.nan)
Wind['Kp_Vy, km/s'] = Wind['Kp_Vy, km/s'].replace(99999.900000, np.nan)
Wind['KP_Vz, km/s'] = Wind['KP_Vz, km/s'].replace(99999.900000, np.nan)
Wind['KP_Speed, km/s'] = Wind['KP_Speed, km/s'].replace(99999.900000, np.nan)
Wind['Kp_proton Density, n/cc'] = Wind['Kp_proton Density, n/cc'].replace(999.990000, np.nan)
Wind['Wind, Xgse,Re'] = Wind['Wind, Xgse,Re'].replace(9999.990000, np.nan)
Wind['Wind, Ygse,Re'] = Wind['Wind, Ygse,Re'].replace(9999.990000, np.nan)
Wind['Wind, Zgse,Re'] = Wind['Wind, Zgse,Re'].replace(9999.990000, np.nan)

Wind['Wind, Xgse,Re'] = Wind['Wind, Xgse,Re']*Re
Wind['Wind, Ygse,Re'] = Wind['Wind, Ygse,Re']*Re
Wind['Wind, Zgse,Re'] = Wind['Wind, Zgse,Re']*Re


# Interpolate NaN values
for column in Wind.columns:
    Wind[column] = Wind[column].interpolate()
    
# Define the desired order of columns as required for Ned's function: WIND
desired_columns_order = [
    'Time',
    'KP_Vx,km/s',
    'Kp_Vy, km/s',
    'KP_Vz, km/s',
    'Wind, Xgse,Re',
    'Wind, Ygse,Re',
    'Wind, Zgse,Re',
    'BZ, GSE, nT',
    'KP_Speed, km/s',
    'Kp_proton Density, n/cc'
]

# Select only the desired columns and reorder them
Wind = Wind[desired_columns_order]


# Drop the original columns
Wind = Wind.drop(['Kp_proton Density, n/cc', 'KP_Speed, km/s', 'BZ, GSE, nT'], axis=1)

for column in ACE.columns:
    ACE[column] = ACE[column].interpolate()
    
# Define the desired order of columns as required for Ned's function: ACE
desired_columns_order = [
    'Time',
    'vx',
    'vy',
    'vz',
    'x',
    'y',
    'z',
    'Bz',
    'n'
]

# Select only the desired columns and reorder them
ACE = ACE[desired_columns_order]

# Need a velocity magnitude column for ACE

# Drop the original columns
ACE.drop(['Bz', 'n'], axis=1, inplace=True)




DSCOVR.rename(columns={'Vx Velocity,km/s': 'vx',
                       'Vy Velocity, km/s': 'vy',
                       'Vz Velocity, km/s':'vz',
                       'Wind, Xgse,Re': 'x',
                       'Wind, Ygse,Re': 'y',
                       'Wind, Zgse,Re': 'z'}, inplace=True)
Wind.rename(columns={'KP_Vx,km/s': 'vx',
                     'Kp_Vy, km/s': 'vy',
                     'KP_Vz, km/s': 'vz',
                     'Wind, Xgse,Re': 'x',
                     'Wind, Ygse,Re': 'y',
                     'Wind, Zgse,Re': 'z'}, inplace=True)


#%% Average velocities: Create 3d array of form:
                                                # [[array of ACE average velocities in each subDF], 
                                                # [array of Wind average velocities in each subDF],
                                                # [array of DSCOVR average velocities in each subDF]]

dataset=[ACE,Wind,DSCOVR]
vAvs=[]
for data in dataset:
    tempVAvs=[]
    for i in range(len(times)):    
        dfFilt = data[(data['Time'] >= times[i][0]) & (data['Time'] <= times[i][1])]
        vAv= np.mean(np.sqrt(np.array(dfFilt['vx'])**2
                                 +np.array(dfFilt['vy'])**2+np.array(dfFilt['vz'])**2))
        tempVAvs.append(vAv)
    vAvs.append(tempVAvs)
    
#%% Average y-z vector offset: Create 3d array of form:
                                                # [[array of ACE average vector [y,z] in each subDF], 
                                                # [array of Wind average vector [y,z] in each subDF],
                                                # [array of DSCOVR average vector [y,z] in each subDF]]

Ds=[]
for i in range(len(times)):
    tempDs=[]
    for data in dataset:
        dfFilt = data[(data['Time'] >= times[i][0]) & (data['Time'] <= times[i][1])]
        pos=[np.mean(dfFilt['y']),np.mean([dfFilt['z']])]
        tempDs.append(pos)
    Ds.append(tempDs)
    
#%% Average y-z magnitude (distance) separation between SC: Create 3d array of form:
    
                                                # [[average ACE-Wind yz separations in each subDF], 
                                                # [average ACE-DSC yz separations in each subDF],
                                                # [average Wind-DSC yz separations in each subDF]]

def calculate_magnitude_difference(coordinates):
    magnitudes = []
    for pair in combinations(coordinates, 2):
        x1, y1 = pair[0]
        x2, y2 = pair[1]
        magnitude = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        magnitudes.append(magnitude)
    return magnitudes

# Example usage
magnitude_differences = [calculate_magnitude_difference(D) for D in Ds]

# So here in 'diffs' we end up with the form described above (at top of this cell)
diffs=np.array(magnitude_differences).T

#%% This is a measure we have created called the 'displacement'
### It is effectively an integral for the negative region of real SYM/H
### Its purpose is to quantify the 'total strength' of the subDF weather and compare to our predictions
### e.g. we want to see if we get better predictions for a larger displacement

symDisp=[]
for i in range(len(times)):
    print(i)
    dfFilt = SYM[(SYM['Time'] >= times[i][0]) & (SYM['Time'] <= times[i][1])]
    symDisp.append(abs(sum( x for x in dfFilt['SYM/H, nT'] if x<0)))

    
#%% This cell returns a list 'ts' which has a value for each subDF which is the mean value of
### the 3 spacecraft's mean time shifts over the subDF's period

tsList=[]
for i in range(len(times)):
    print(i)
    tempDs=[]
    for data in dataset:
        dfFilt = data[(data['Time'] >= times[i][0]) & (data['Time'] <= times[i][1])]
        ts=np.mean((np.array(dfFilt['x'])-10*Re)/np.array(dfFilt['vx']))
        tempDs.append(ts)
    tsList.append(tempDs)

ts=np.mean(tsList,axis=1)

#%% Now we begin plotting - this is the average propoagation time calculated in the cell just above

with plt.style.context('ggplot'):
    
    plt.hist(ts/60,bins=100,color='#1f77b4')

plt.xlabel("Average Propagation Time to Magnetopause (mins)")
plt.ylabel("Frequency")
plt.title("Distribution of Calculated Propagation Times")
plt.show()
        

#%% Here we compare cross correlations for multi vs OMNI predictions in a histogram

with plt.style.context('ggplot'):
    
    plt.hist([multi,OMNIvals])

plt.xlabel("Cross Correlation in SYM/H")
plt.ylabel("Frequency")
#plt.title("Performance Comparison")
plt.legend(["Weighted Average", "OMNI Propagated"])
#combVav=np.mean(vAvs,axis=0)
print(np.mean(ACEvals))
print(np.mean(multi))
print(np.mean(OMNIvals))
plt.show()


#%% Histogram of all 4 max CCs

with plt.style.context('ggplot'):
    # Set the 'tab10' color palette
    sns.set_palette("tab10")

    # Plot the histogram using sns.histplot
    plt.hist([ACEvals, windvals, dscvals, multi], alpha=1)

plt.xlabel("Cross Correlation in SYM/H")
plt.ylabel("Frequency")
#plt.title("Performance Comparison, Max CC")
plt.legend([ "ACE","Wind","DSCOVR","Weighted Average"])

print('Mean Cross Correlations:')
print('ACE:',np.mean(ACEvals))
print('Wind:',np.mean(windvals))
print('DSCOVR:',np.mean(dscvals))
print('Multi:',np.mean(multi))


#%% Spacecraft-spacecraft offset y-z offsets vs the cross correlations between their predictions
### Not sure this is very useful? What does it mean? I guess we should prove that closer spacecraft make more
### similar predictions so we should see a negative gradient - higher yz offset = lower CC

plt.plot(diffs[0]/1000,CCs0,'x')
plt.plot(diffs[1]/1000,CCs1,'x')
plt.plot(diffs[2]/1000,CCs2,'x')
plt.xlabel("Intra Spacecraft Offset, 1000kms")
plt.ylabel("Cross Correlation")
plt.title("Cross Correlation Between SS predictions as a Function of y-z Offset")


#%%

combVav=np.mean(vAvs,axis=0)

def binPlot(x, y, xlab,ylab,title,num_bins=10,colorbar=True):
    # Determine bin edges
    bin_edges = np.linspace(np.min(x), np.max(x), num_bins + 1)

    # Initialize lists to store bin centers, means, and standard errors
    bin_centers = []
    mean_values = []
    std_errors = []
    freqs=[]
    min_points_per_bin=40

    # Iterate over each bin
    for i in range(num_bins):
        # Identify data points within the current bin
        bin_mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        
        bin_x = x[bin_mask]
        bin_y = y[bin_mask]
        if len(bin_y) < min_points_per_bin:
            continue
        freqs.append(np.count_nonzero(bin_mask))
        # Calculate mean and standard error of y within the bin
        mean_y = np.mean(bin_y)
        std_error_y = np.std(bin_y) / np.sqrt(len(bin_y))

        # Append bin center, mean, and standard error to lists
        bin_centers.append(np.mean(bin_x))
        mean_values.append(mean_y)
        std_errors.append(std_error_y)
     # Plot the means with error bars
    if colorbar:
        plt.scatter(bin_centers,mean_values,c=freqs,cmap='viridis')
        cbar = plt.colorbar(label='Color')
        cbar.set_label('# of values in bin')
        plt.errorbar(bin_centers, mean_values, yerr=std_errors, fmt='none',color='black')
    else:
        plt.errorbar(bin_centers, mean_values, yerr=std_errors, fmt='o')    
    
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)

    #plt.show()
    
    return [np.array(lst) for lst in [bin_centers,mean_values,std_errors]]

def linFit(x, y, y_err):
    """
    Fit a straight line to data with uncertainties and plot the result.

    Parameters:
        x (array-like): The x data.
        y (array-like): The y data.
        y_err (array-like): The uncertainties in y data.
    """
    # Define the linear function to fit
    def linear_func(x, m, c):
        return m * x + c

    # Perform the curve fitting
    popt, pcov = curve_fit(linear_func, x, y, sigma=y_err)

    # Calculate the residuals
    residuals = y - linear_func(x, *popt)

    # Calculate the reduced chi-square
    goodness_of_fit = np.sum((residuals / y_err)**2) / (len(x) - 2)

    # Plot the data with error bars


    # Plot the fitted line
    plt.plot(x, linear_func(x, *popt), color='blue', label='Fitted Line')


    print("Optimal parameters (m, c):", popt)
    print("Covariance matrix:", pcov)
    print("Goodness of fit (reduced chi-square):", goodness_of_fit)

#binPlot(combVav,multi,"Solar Wind Velocity, km/s","Mean CC", "CC against $v_{SW}$",10)
binPlot(diffs[0]/1000,CCs0,"Intra Spacecraft Offset", "Mean CC", "CC against Offset",10,False)
binPlot(diffs[1]/1000,CCs1,"Intra Spacecraft Offset", "Mean CC", "CC against Offset",10,False)
binPlot(diffs[2]/1000,CCs2,"Intra Spacecraft Offset (1000kms)", "Mean CC", "CC against Offset",10,False)
plt.legend(["ACE, Wind","ACE, DSCOVR","Wind,DSCOVR"])
plt.grid(alpha=0.5)
#ACE and Wind, ACE and DSCOVR and Wind and DSCOVR
plt.show()
avdiffs=np.mean(diffs,axis=0)
weightedDiff=binPlot(avdiffs/1000,multi,"Average Spacecraft Offset, (1000kms)", "Mean CC in Bin",
                     "CC between Weighted Average and Real SYM/H against Average S/C Offset",15)
linFit(*weightedDiff)
plt.grid(alpha=0.5)
plt.show()
plt.hist([CCs0,CCs1,CCs2])
plt.legend(["ACE, Wind","ACE, DSCOVR","Wind, DSCOVR"])
plt.xlabel("Cross Correlation Between Pairs of SS Predictions")
plt.ylabel("Number")
plt.show()


#%%

weightedV=binPlot(combVav,multi,"Average $v_{SW}$","Mean CC in Bin","Weighted Average CC against $v_{SW}$ with min. Occupancy",0,True)
plt.ylim(0.15,0.45)
linFit(*weightedV)
plt.grid(alpha=0.5)
plt.show()

#%%

plt.hist(deltaT/60,bins=10)
plt.xlabel("$\Delta t$, mins")
plt.ylabel("Number")
plt.title("Distribution of Time Errors")
print(np.mean(deltaT))
plt.show()

binPlot(abs(deltaT/60),multi,"Absolute Time Shift Errors, mins", "Max CC", "Maximum CC against Absolute Shift Error",20)
plt.grid(alpha=0.5)

#%%

binPlot(np.array(symDisp),np.array(multi),'SYM/H Displacement', "Average CC", "Average CC against Displacement",5)
plt.show()
plt.hist(symDisp)
plt.xlabel("'Displacement'")
plt.ylabel("Number")
plt.title("Distribution of Period Eventfulness")
plt.xlim(0,15000)
#%%

def binThresh(x, y, threshold, xlab,ylab,title,num_bins=10, min_points_per_bin=50):
    # Filter data based on threshold
    filtered_indices = np.where(x < threshold)[0]
    x_filtered = x[filtered_indices]
    y_filtered = y[filtered_indices]

    # Determine bin edges
    bin_edges = np.linspace(np.min(x_filtered), np.max(x_filtered), num_bins + 1)

    # Initialize lists to store bin centers, means, and standard errors
    bin_centers = []
    mean_values = []
    std_errors = []

    # Iterate over each bin
    for i in range(num_bins):
        # Identify data points within the current bin
        bin_mask = (x_filtered >= bin_edges[i]) & (x_filtered < bin_edges[i + 1])
        bin_x = x_filtered[bin_mask]
        bin_y = y_filtered[bin_mask]

        # Check if there are sufficient points in the bin
        if len(bin_y) < min_points_per_bin:
            continue

        # Calculate mean and standard error of y within the bin
        mean_y = np.mean(bin_y)
        std_error_y = np.std(bin_y) / np.sqrt(len(bin_y))

        # Append bin center, mean, and standard error to lists
        bin_centers.append(np.mean(bin_x))
        mean_values.append(mean_y)
        std_errors.append(std_error_y)

    # Plot the means with error bars
    plt.errorbar(bin_centers, mean_values, yerr=std_errors, fmt='o')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.show()


binThresh(np.array(symDisp),np.array(multi),5000,"Displacement","$CC_{WA}$","CC against 'Displacement'")
plt.show()
binThresh(np.array(symDisp),np.array(abs(deltaT))/60,5000,"Displacement","$\Delta$t, minutes","$\Delta$t against 'Displacement")
plt.show()
plt.plot(np.array(symDisp),np.array(abs(deltaT))/60,'x')
plt.xlabel("'Displacement'")
plt.ylabel("$\Delta$t, minutes")


#%%

lims=[(min([min(ACEvals),min(multi)])),max([max(ACEvals),max(multi)])]

yx=np.linspace(lims[0],lims[1],100)
plt.plot(ACEvals,multi,'x')
plt.plot(yx,yx,linestyle='dashed')
plt.xlabel("$CC_{ACE}$")
plt.ylabel("$CC_{WA}$")


#%%

binPlot(multi,ACEvals,"$CC_{WA}$","$CC_{ACE}$","CC Comparison",20)
plt.plot(yx,yx,linestyle='dashed')


#%%

mask= abs(deltaT)>2400
print(len(multi)-sum(mask))
binThresh(np.array(symDisp)[mask],np.array(multi)[mask],5000,"Displacement","$CC_{WA}$","CC against 'Displacement'")
plt.show()
binThresh(np.array(symDisp)[mask],(np.array(abs(deltaT))[mask])/60,5000,"Displacement","$\Delta$t, minutes","$\Delta$t against 'Displacement")


#%%

mask=abs(deltaT)<2400
modMult=multi[mask]
plt.hist(modMult)
#plt.legend(["Multi","Filtered Multi"])
plt.xlabel("CC")
plt.ylabel("Frequency")
print(np.mean(multi))
print(np.mean(modMult))
plt.title("Filtered Max CCs")


#%%

binPlot(abs(np.array(deltaT)), np.array(symDisp), "deltaT", "disp", "title")

#%%

temporary=SYM[4000:4000+60*4]