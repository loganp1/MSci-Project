# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 17:01:14 2024

@author: logan
"""

import numpy as np
from single_spacecraft_shift import singleSpacecraft_propagate


def EP_singleSC_propagator(df,spacecraft_name):
    
    # Define P & E functions
    def P(proton_density,velocity_mag):
        return proton_density*velocity_mag**2*2e-6
    def E(velocity_mag,Bz):
        return -velocity_mag*Bz*1e-3
    
    # Need coordinates in km not Earth raddi Re as they are for DSCOVR and Wind ONLY
    Re = 6371
    
    # DSCOVR
    # Identify rows with NaN values from USEFUL columns
    if spacecraft_name == 'DSCOVR':
        
        df['Field Magnitude,nT'] = df['Field Magnitude,nT'].replace(9999.99, np.nan)
        df['Vector Mag.,nT'] = df['Vector Mag.,nT'].replace(9999.99, np.nan)
        df['BX, GSE, nT'] = df['BX, GSE, nT'].replace(9999.99, np.nan)
        df['BY, GSE, nT'] = df['BY, GSE, nT'].replace(9999.99, np.nan)
        df['BZ, GSE, nT'] = df['BZ, GSE, nT'].replace(9999.99, np.nan)
        df['Speed, km/s'] = df['Speed, km/s'].replace(99999.9, np.nan)
        df['Vx Velocity,km/s'] = df['Vx Velocity,km/s'].replace(99999.9, np.nan)
        df['Vy Velocity, km/s'] = df['Vy Velocity, km/s'].replace(99999.9, np.nan)
        df['Vz Velocity, km/s'] = df['Vz Velocity, km/s'].replace(99999.9, np.nan)
        df['Proton Density, n/cc'] = df['Proton Density, n/cc'].replace(999.999, np.nan)
        df['Wind, Xgse,Re'] = df['Wind, Xgse,Re'].replace(9999.99, np.nan)
        df['Wind, Ygse,Re'] = df['Wind, Ygse,Re'].replace(9999.99, np.nan)
        df['Wind, Zgse,Re'] = df['Wind, Zgse,Re'].replace(9999.99, np.nan)
        
        # Have to do change to km AFTER removing vals else fill value will change!
        df['Wind, Xgse,Re'] = df['Wind, Xgse,Re']*Re
        df['Wind, Ygse,Re'] = df['Wind, Ygse,Re']*Re
        df['Wind, Zgse,Re'] = df['Wind, Zgse,Re']*Re
        
        # Interpolate NaN values
        for column in df.columns:
            df[column] = df[column].interpolate()
            
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
        df = df[desired_columns_order]
        
        df['P'] = P(df['Proton Density, n/cc'], df['Speed, km/s'])
        df['E'] = E(df['Speed, km/s'], df['BZ, GSE, nT'])
        
        # Drop the original columns
        df.drop(['Proton Density, n/cc', 'Speed, km/s', 'BZ, GSE, nT'], axis=1, inplace=True)
        
    # WIND
    # Identify rows with NaN values from USEFUL columns
    if spacecraft_name == 'Wind':
        
        df['BX, GSE, nT'] = df['BX, GSE, nT'].replace(9999.990000, np.nan)
        df['BY, GSE, nT'] = df['BY, GSE, nT'].replace(9999.990000, np.nan)
        df['BZ, GSE, nT'] = df['BZ, GSE, nT'].replace(9999.990000, np.nan)
        df['Vector Mag.,nT'] = df['Vector Mag.,nT'].replace(9999.990000, np.nan)
        df['Field Magnitude,nT'] = df['Field Magnitude,nT'].replace(9999.990000, np.nan)
        df['KP_Vx,km/s'] = df['KP_Vx,km/s'].replace(99999.900000, np.nan)
        df['Kp_Vy, km/s'] = df['Kp_Vy, km/s'].replace(99999.900000, np.nan)
        df['KP_Vz, km/s'] = df['KP_Vz, km/s'].replace(99999.900000, np.nan)
        df['KP_Speed, km/s'] = df['KP_Speed, km/s'].replace(99999.900000, np.nan)
        df['Kp_proton Density, n/cc'] = df['Kp_proton Density, n/cc'].replace(999.990000, np.nan)
        df['Wind, Xgse,Re'] = df['Wind, Xgse,Re'].replace(9999.990000, np.nan)
        df['Wind, Ygse,Re'] = df['Wind, Ygse,Re'].replace(9999.990000, np.nan)
        df['Wind, Zgse,Re'] = df['Wind, Zgse,Re'].replace(9999.990000, np.nan)
        
        df['Wind, Xgse,Re'] = df['Wind, Xgse,Re']*Re
        df['Wind, Ygse,Re'] = df['Wind, Ygse,Re']*Re
        df['Wind, Zgse,Re'] = df['Wind, Zgse,Re']*Re
        
        
        # Interpolate NaN values
        for column in df.columns:
            df[column] = df[column].interpolate()
            
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
        df = df[desired_columns_order]
        
        df['P'] = P(df['Kp_proton Density, n/cc'], df['KP_Speed, km/s'])
        df['E'] = E(df['KP_Speed, km/s'], df['BZ, GSE, nT'])
        
        # Drop the original columns
        df.drop(['Kp_proton Density, n/cc', 'KP_Speed, km/s', 'BZ, GSE, nT'], axis=1, inplace=True)
        
    # ACE (already cleaned)
    # Interpolate NaN values
    if spacecraft_name == 'ACE':
        
        for column in df.columns:
            df[column] = df[column].interpolate()
            
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
        df = df[desired_columns_order]
        
        # Need a velocity magnitude column for ACE
        df['v'] = np.sqrt(df['vx']**2+df['vy']**2+df['vz']**2)
        df['P'] = P(df['n'], df['v'])
        df['E'] = E(df['v'], df['Bz'])
        
        # Drop the original columns
        df.drop(['Bz', 'v', 'n'], axis=1, inplace=True)
    
    
    # Transform the data to desired format
    array = df.to_numpy().T
    print(min(array[4]))
    
    # Chosen BSN location at (X=14,Y=0,Z=0)
    
    times_BSN = np.zeros(len(df))  # Don't think times matter at BSN, its always same in our model!
    BSN_x = np.full(len(df),14*Re)
    BSN_y = np.full(len(df),0)
    BSN_z = np.full(len(df),0)
    
    array_BSN = np.array([times_BSN,BSN_x,BSN_y,BSN_z])

    # Perform propagation
    return singleSpacecraft_propagate(array,array_BSN)