# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:47:26 2024

@author: logan
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:58:51 2024

@author: logan
"""

import numpy as np
from weighted_average_shift import weightedAv


def EP_weightedAv_propagator(df1,df2,df3):

    
    # DSCOVR
    # Identify rows with NaN values from USEFUL columns
    df1['BZ, GSE, nT'] = df1['BZ, GSE, nT'].replace(9999.990, np.nan)
    df1['Speed, km/s'] = df1['Speed, km/s'].replace(99999.900, np.nan)
    df1['Proton Density, n/cc'] = df1['Proton Density, n/cc'].replace(999.999, np.nan)
    
    # Interpolate NaN values
    for column in df1.columns:
        df1[column] = df1[column].interpolate()
        
    # WIND
    # Identify rows with NaN values from USEFUL columns
    df2['BZ, GSE, nT'] = df2['BZ, GSE, nT'].replace(9999.990, np.nan)
    df2['KP_Speed, km/s'] = df2['KP_Speed, km/s'].replace(99999.900, np.nan)
    df2['Kp_proton Density, n/cc'] = df2['Kp_proton Density, n/cc'].replace(999.99, np.nan)
    
    # Interpolate NaN values
    for column in df2.columns:
        df2[column] = df2[column].interpolate()
        
     # ACE (already cleaned)
    # Interpolate NaN values
    for column in df3.columns:
        df3[column] = df3[column].interpolate()
    
   
    
    # Define the desired order of columns as required for Ned's function: DSCOVR
    desired_columns_order1 = [
        'Time',
        'Wind, Xgse,Re',
        'Wind, Ygse,Re',
        'Wind, Zgse,Re',
        'Vx Velocity,km/s',
        'Vy Velocity, km/s',
        'Vz Velocity, km/s',
        'BZ, GSE, nT',
        'Speed, km/s',
        'Proton Density, n/cc'
    ]
    
    
    # Define the desired order of columns as required for Ned's function: WIND
    desired_columns_order2 = [
        'Time',
        'Wind, Xgse,Re',
        'Wind, Ygse,Re',
        'Wind, Zgse,Re',
        'KP_Vx,km/s',
        'Kp_Vy, km/s',
        'KP_Vz, km/s',
        'BZ, GSE, nT',
        'KP_Speed, km/s',
        'Kp_proton Density, n/cc'
    ]
    
    
    # Define the desired order of columns as required for Ned's function: ACE
    desired_columns_order3 = [
        'Time',
        'x',
        'y',
        'z',
        'vx',
        'vy',
        'vz',
        'Bz',
        'n'
    ]
    
    
    # Select only the desired columns and reorder them
    df1 = df1[desired_columns_order1]
    df2 = df2[desired_columns_order2]
    df3 = df3[desired_columns_order3]
    
    def P(proton_density,velocity_mag):
        # Found the correct unit scaling on OMNIweb using units data is given in!
        return proton_density*velocity_mag**2*2e-6
    
    def E(velocity_mag,Bz):
        
        return -velocity_mag*Bz*1e-3
    
    # Need a velocity magnitude column for ACE
    df3['v'] = np.sqrt(df3['vx']**2+df3['vy']**2+df3['vz']**2)
    
    df1['P'] = P(df1['Proton Density, n/cc'], df1['Speed, km/s'])
    df1['E'] = E(df1['Speed, km/s'], df1['BZ, GSE, nT'])
    
    df2['P'] = P(df2['Kp_proton Density, n/cc'], df2['KP_Speed, km/s'])
    df2['E'] = E(df2['KP_Speed, km/s'], df2['BZ, GSE, nT'])
    
    df3['P'] = P(df3['n'], df3['v'])
    df3['E'] = E(df3['v'], df3['Bz'])
    
    # Drop the original columns
    df1.drop(['Proton Density, n/cc', 'Speed, km/s', 'BZ, GSE, nT'], axis=1, inplace=True)
    df2.drop(['Kp_proton Density, n/cc', 'KP_Speed, km/s', 'BZ, GSE, nT'], axis=1, inplace=True)
    df3.drop(['Bz', 'v', 'n'], axis=1, inplace=True)
    
    
    # Transform the data to desired format
    array1 = df1.to_numpy().T
    array2 = df2.to_numpy().T
    array3 = df3.to_numpy().T
    
   
    
    # Chosen BSN location at (X=14,Y=0,Z=0) units in Re
    
    ####
    #### IMPORTANT!!! Surely results wrong as velocity in km/s but distance in Re so time shift WRONG!!!!!!!!
    ####
    
    times_BSN = np.zeros(len(df1))  # Don't think times matter at BSN, its always same in our model!
    BSN_x = np.full(len(df1),14)
    BSN_y = np.full(len(df1),0)
    BSN_z = np.full(len(df1),0)
    
    array_BSN = np.array([times_BSN,BSN_x,BSN_y,BSN_z])
    
    slist = [array1,array2,array3]
    
    # Perform propagation
    
    return weightedAv(slist,array_BSN,8), weightedAv(slist,array_BSN,7)



