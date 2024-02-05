# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:44:58 2024

@author: logan
"""

import numpy as np
import pandas as pd

class SC_Propagation:
    
    ''' 
    
    This class serves to clean the data and leave it in the required format for the single spacecraft and 
    multi-spacecraft propagations. The user can choose whether they would like to perform a single sc shift
    or one of (hopefully) a few multi-spacecraft shift methods. These propagation functions will return 
    dataframes with the downstream time series along with the shifted data to match the time series.
    
    The class MUST be initialised as a dictionary as follows:
    
    instance_of_class_name = SC_Propagation({SC1_name: SC1_data, SC2_name: SC2_data, SC3_name: SC3_data]})
    
    - Where SCi_data are uploaded as pandas dataframes pd.read_csv()
    
    '''
    
    def __init__(self, SCdict):
        
        self._ACE = SCdict['ACE']
        self._DSCOVR = SCdict['DSCOVR']
        self._Wind = SCdict['Wind']
        
        # Define conversion factor from Earth radii to km
        self._Re = 6371     
        

    @property # Do this to update the dictionary with the cleaned dataframes
    def _SCdict(self):
        return {'ACE': self._ACE, 'DSCOVR': self._DSCOVR, 'Wind': self._Wind}
    
        
    @property # This allows us to access self._multiSC as if it's an attribute - it will always be updated
    def _multiSC(self):
        return [self._ACE.to_numpy().T, self._DSCOVR.to_numpy().T, self._Wind.to_numpy().T]
        
  

    def P(self,proton_density,velocity_mag):
        return proton_density*velocity_mag**2*2e-6
    
    def E(self,velocity_mag,Bz):
        return -velocity_mag*Bz*1e-3 
    
    
    def getDFs(self, sc_name):
        
        if sc_name == 'ACE':
            return self._ACE
        
        if sc_name == 'DSCOVR':
            return self._DSCOVR
        
        if sc_name == 'Wind':
            return self._Wind
        
        
    def required_form(self):
    
            
        # DSCOVR
        self._DSCOVR['Field Magnitude,nT'] = self._DSCOVR['Field Magnitude,nT'].replace(9999.99, np.nan)
        self._DSCOVR['Vector Mag.,nT'] = self._DSCOVR['Vector Mag.,nT'].replace(9999.99, np.nan)
        self._DSCOVR['BX, GSE, nT'] = self._DSCOVR['BX, GSE, nT'].replace(9999.99, np.nan)
        self._DSCOVR['BY, GSE, nT'] = self._DSCOVR['BY, GSE, nT'].replace(9999.99, np.nan)
        self._DSCOVR['BZ, GSE, nT'] = self._DSCOVR['BZ, GSE, nT'].replace(9999.99, np.nan)
        self._DSCOVR['Speed, km/s'] = self._DSCOVR['Speed, km/s'].replace(99999.9, np.nan)
        self._DSCOVR['Vx Velocity,km/s'] = self._DSCOVR['Vx Velocity,km/s'].replace(99999.9, np.nan)
        self._DSCOVR['Vy Velocity, km/s'] = self._DSCOVR['Vy Velocity, km/s'].replace(99999.9, np.nan)
        self._DSCOVR['Vz Velocity, km/s'] = self._DSCOVR['Vz Velocity, km/s'].replace(99999.9, np.nan)
        self._DSCOVR['Proton Density, n/cc'] = self._DSCOVR['Proton Density, n/cc'].replace(999.999, np.nan)
        self._DSCOVR['Wind, Xgse,Re'] = self._DSCOVR['Wind, Xgse,Re'].replace(9999.99, np.nan)
        self._DSCOVR['Wind, Ygse,Re'] = self._DSCOVR['Wind, Ygse,Re'].replace(9999.99, np.nan)
        self._DSCOVR['Wind, Zgse,Re'] = self._DSCOVR['Wind, Zgse,Re'].replace(9999.99, np.nan)
        
        # Have to do change to km AFTER removing vals else fill value will change!
        self._DSCOVR['Wind, Xgse,Re'] = self._DSCOVR['Wind, Xgse,Re']*self._Re
        self._DSCOVR['Wind, Ygse,Re'] = self._DSCOVR['Wind, Ygse,Re']*self._Re
        self._DSCOVR['Wind, Zgse,Re'] = self._DSCOVR['Wind, Zgse,Re']*self._Re
        
        # Interpolate NaN values
        for column in self._DSCOVR.columns:
            self._DSCOVR[column] = self._DSCOVR[column].interpolate()
            
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
        self._DSCOVR = self._DSCOVR[desired_columns_order]
        
        self._DSCOVR['P'] = self.P(self._DSCOVR['Proton Density, n/cc'], self._DSCOVR['Speed, km/s'])
        self._DSCOVR['E'] = self.E(self._DSCOVR['Speed, km/s'], self._DSCOVR['BZ, GSE, nT'])
        
        # Drop the original columns
        self._DSCOVR.drop(['Proton Density, n/cc', 'Speed, km/s', 'BZ, GSE, nT'], axis=1, inplace=True)
        
            
        # WIND
            
        self._Wind['BX, GSE, nT'] = self._Wind['BX, GSE, nT'].replace(9999.990000, np.nan)
        self._Wind['BY, GSE, nT'] = self._Wind['BY, GSE, nT'].replace(9999.990000, np.nan)
        self._Wind['BZ, GSE, nT'] = self._Wind['BZ, GSE, nT'].replace(9999.990000, np.nan)
        self._Wind['Vector Mag.,nT'] = self._Wind['Vector Mag.,nT'].replace(9999.990000, np.nan)
        self._Wind['Field Magnitude,nT'] = self._Wind['Field Magnitude,nT'].replace(9999.990000, np.nan)
        self._Wind['KP_Vx,km/s'] = self._Wind['KP_Vx,km/s'].replace(99999.900000, np.nan)
        self._Wind['Kp_Vy, km/s'] = self._Wind['Kp_Vy, km/s'].replace(99999.900000, np.nan)
        self._Wind['KP_Vz, km/s'] = self._Wind['KP_Vz, km/s'].replace(99999.900000, np.nan)
        self._Wind['KP_Speed, km/s'] = self._Wind['KP_Speed, km/s'].replace(99999.900000, np.nan)
        self._Wind['Kp_proton Density, n/cc'] = self._Wind['Kp_proton Density, n/cc'].replace(999.990000, np.nan)
        self._Wind['Wind, Xgse,Re'] = self._Wind['Wind, Xgse,Re'].replace(9999.990000, np.nan)
        self._Wind['Wind, Ygse,Re'] = self._Wind['Wind, Ygse,Re'].replace(9999.990000, np.nan)
        self._Wind['Wind, Zgse,Re'] = self._Wind['Wind, Zgse,Re'].replace(9999.990000, np.nan)
        
        self._Wind['Wind, Xgse,Re'] = self._Wind['Wind, Xgse,Re']*self._Re
        self._Wind['Wind, Ygse,Re'] = self._Wind['Wind, Ygse,Re']*self._Re
        self._Wind['Wind, Zgse,Re'] = self._Wind['Wind, Zgse,Re']*self._Re
        
        
        # Interpolate NaN values
        for column in self._Wind.columns:
            self._Wind[column] = self._Wind[column].interpolate()
            
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
        self._Wind = self._Wind[desired_columns_order]
        
        self._Wind['P'] = self.P(self._Wind['Kp_proton Density, n/cc'], self._Wind['KP_Speed, km/s'])
        self._Wind['E'] = self.E(self._Wind['KP_Speed, km/s'], self._Wind['BZ, GSE, nT'])
        
        # Drop the original columns
        self._Wind.drop(['Kp_proton Density, n/cc', 'KP_Speed, km/s', 'BZ, GSE, nT'], axis=1, inplace=True)
            
        # ACE (already cleaned)
        # Interpolate NaN values

        for column in self._ACE.columns:
            self._ACE[column] = self._ACE[column].interpolate()
            
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
        self._ACE = self._ACE[desired_columns_order]
        
        # Need a velocity magnitude column for ACE
        self._ACE['v'] = np.sqrt(self._ACE['vx']**2+self._ACE['vy']**2+self._ACE['vz']**2)
        self._ACE['P'] = self.P(self._ACE['n'], self._ACE['v'])
        self._ACE['E'] = self.E(self._ACE['v'], self._ACE['Bz'])
        
        # Drop the original columns
        self._ACE.drop(['Bz', 'v', 'n'], axis=1, inplace=True)
        
    
    
    def singleSC_Propagate(self, sc_name):
        
        '''
        
        Function that takes inidividual spacecraft data from L1 [s], and the location we are propagating to 
        [sRef], and returns propagated data with its corresponding time series.
        
        INPUTS:
        - s : list/array
        Spacecraft data: [[t0,vx0,vy0,vz0,x0,y0,z0,DATA],[t1,vx1,vy1,vz1,x1,y1,z1,...].T (so this form transposed)

        - sRef: list/array                      
        Target: [[x0,y0,z0,DATA],[...].T] (so this form transposed)
        
        RETURNS:
        - s[0] The new time series at the desired position propagated to
        - s[8] Propagated electric field data as we will input this as the 9th column = 8th index in s
        - s[7] Propagated pressure data as we will input this as the 8th column = 7th index in s
        - Ts   Time shifts for each data point, useful for histograms and comparing to SW velocity histograms
        
        '''
        
        # Transform the data to desired format
        s = self._SCdict[sc_name].to_numpy().T
        
        # Chosen target (BSN) location at (X=14,Y=0,Z=0)
        BSN_x = np.full(len(self._ACE),14*self._Re)
        BSN_y = np.full(len(self._ACE),0)
        BSN_z = np.full(len(self._ACE),0)
        
        array_BSN = np.array([BSN_x,BSN_y,BSN_z])
        
        # vxAv=np.mean(s[1]) # These have potential for us to not just use the instantaneous velocity for time
        # vyAv=np.mean(s[2]) # shift calculations, but an average over a certain window
        # vzAv=np.mean(s[3])
        Ts=(abs(array_BSN[0]-s[4]))/(abs(s[1]))
        s=[np.asarray(s[0])+np.asarray(Ts),s[1],s[2],s[3],np.array(s[4])+s[1]*Ts,
                                  np.array(s[5])+s[2]*Ts,
                                  np.array(s[6])+s[3]*Ts,s[7],s[8]]
        
        return pd.DataFrame({'Time': s[0], 'Efield': s[8], 'Pressure': s[7], 'Time Shifts': Ts})
    
    

    def WA_method(self,sList,indS):
        
        """
        
        Function that takes multiple spacecraft data from L1 [sList], and the location we are propagating to 
        [sRef], and returns propagated data with its corresponding time series using the weighted average method.

        INPUTS:
        - sList : list/array
        Spacecraft data: [[t0,vx0,vy0,vz0,x0,y0,z0,DATA],[t1,vx1,vy1,vz1,x1,y1,z1,...],...].T 
            
        - sRef: list/array                      
        Target: [[x0,y0,z0],[x1,y1,z1],...].T
        
        indS: int
        Index of data you want to average in s
        
        RETURNS:
        - weighted    

        """
        
        # Chosen target (BSN) location at (X=14,Y=0,Z=0)
        BSN_x = np.full(len(self._ACE),14*self._Re)
        BSN_y = np.full(len(self._ACE),0)
        BSN_z = np.full(len(self._ACE),0)
        
        array_BSN = np.array([BSN_x,BSN_y,BSN_z])
        
        def getWeights(ri):
            r0=50*6378
            return np.exp(-1*ri/r0)
        
        weights=[]
        for s in sList:
            Ts=(array_BSN[0]-s[4])/(s[1])
            s=[s[0]+Ts,s[1],s[2],s[3],np.array(s[4])+s[1]*Ts,
                                       np.array(s[5])+s[2]*Ts,
                                       np.array(s[6])+s[3]*Ts,
                                       s[7],s[8]]
            deltay=s[5]-array_BSN[1]
            deltaz=s[6]-array_BSN[2]
            offsets=np.sqrt(deltay**2+deltaz**2)
            weights.append(np.array([getWeights(ri) for ri in offsets]))
            
        weights=np.array(weights).T  
        weighted_qtty=[]
        
        for i in range(0,len(sList[0].T)):
            tempBz=[]
            for data in sList:
                tempBz.append(data[indS][i])
            B=sum(np.array(tempBz)*weights[i])/sum(weights[i])
            weighted_qtty.append(B)
        
        shifted_time_series = np.asarray(s[0])      
        return weighted_qtty, shifted_time_series
    
    
    def multiSC_WA_Propagate(self):
        
        new_E, new_time = self.WA_method(self._multiSC,8)
        new_P, new_time = self.WA_method(self._multiSC,7)
        
        return pd.DataFrame({'Time': new_time, 'Efield': new_E, 'Pressure': new_P})
        
        