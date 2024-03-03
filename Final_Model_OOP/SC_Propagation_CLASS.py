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
        self._Re = 6378    
        

    @property # Do this to update the dictionary with the cleaned dataframes
    def _SCdict(self):
        return {'ACE': self._ACE, 'DSCOVR': self._DSCOVR, 'Wind': self._Wind}
    
        
    @property # This allows us to access self._multiSC as if it's an attribute - it will always be updated
    def _multiSC(self):
        return [self._ACE.to_numpy().T, self._DSCOVR.to_numpy().T, self._Wind.to_numpy().T]
        
  

    def P(self,proton_density,velocity_mag):
        return proton_density*velocity_mag**2*2e-6
    
    def E(self,velocity_mag,Bz):
        return -velocity_mag*Bz*2e-3 
    
    
    def getDFs(self, sc_name):
        
        if sc_name == 'ACE':
            return self._ACE
        
        if sc_name == 'DSCOVR':
            return self._DSCOVR
        
        if sc_name == 'Wind':
            return self._Wind
        
        
    def required_form(self, keep_primary_data=False):
    
            
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
        
        self._DSCOVR = self._DSCOVR.copy()
        self._DSCOVR.loc[:, 'P'] = self.P(self._DSCOVR['Proton Density, n/cc'], self._DSCOVR['Speed, km/s'])
        self._DSCOVR.loc[:, 'E'] = self.E(self._DSCOVR['Speed, km/s'], self._DSCOVR['BZ, GSE, nT'])
        
        # Drop the original if keep_primary_data = False (or unspecified as auto set to this)
        if keep_primary_data == False:
            self._DSCOVR = self._DSCOVR.drop(['Proton Density, n/cc', 'Speed, km/s', 'BZ, GSE, nT'], axis=1)

        
            
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
        
        self._Wind = self._Wind.copy()
        self._Wind['P'] = self.P(self._Wind['Kp_proton Density, n/cc'], self._Wind['KP_Speed, km/s'])
        self._Wind['E'] = self.E(self._Wind['KP_Speed, km/s'], self._Wind['BZ, GSE, nT'])
        
        # Drop the original columns
        if keep_primary_data == False:
            self._Wind = self._Wind.drop(['Kp_proton Density, n/cc', 'KP_Speed, km/s', 'BZ, GSE, nT'], axis=1)
            
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
        self._ACE = self._ACE.copy()
        self._ACE['v'] = np.sqrt(self._ACE['vx']**2+self._ACE['vy']**2+self._ACE['vz']**2).values
        self._ACE['P'] = self.P(self._ACE['n'], self._ACE['v'])
        self._ACE['E'] = self.E(self._ACE['v'], self._ACE['Bz'])
        
        # Drop the original columns
        if keep_primary_data == False:
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
        BSN_x = np.full(len(self._ACE),10*self._Re)
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
    
    
    
    def getWeights(self,ri):
        r0=50*6378
        return np.exp(-1*ri/r0)
    

    # def WA_method(self,sList,indS):
        
    #     """
        
    #     Function that takes multiple spacecraft data from L1 [sList], and the location we are propagating to 
    #     [sRef], and returns propagated data with its corresponding time series using the weighted average method.

    #     INPUTS:
    #     - sList : list/array
    #     Spacecraft data: [[t0,vx0,vy0,vz0,x0,y0,z0,DATA],[t1,vx1,vy1,vz1,x1,y1,z1,...],...].T 
            
    #     - sRef: list/array                      
    #     Target: [[x0,y0,z0],[x1,y1,z1],...].T
        
    #     indS: int
    #     Index of data you want to average in s
        
    #     RETURNS:
    #     - weighted    

    #     """

    #     # Chosen target (BSN) location at (X=14,Y=0,Z=0)
    #     BSN_x = np.full(len(self._ACE),14*self._Re)
    #     BSN_y = np.full(len(self._ACE),0)
    #     BSN_z = np.full(len(self._ACE),0)
        
    #     array_BSN = np.array([BSN_x,BSN_y,BSN_z])
        
        
    #     weights=[]
    #     for s in sList:
    #         Ts=(array_BSN[0]-s[4])/(s[1])
    #         s=[s[0]+Ts,s[1],s[2],s[3],np.array(s[4])+s[1]*Ts,
    #                                    np.array(s[5])+s[2]*Ts,
    #                                    np.array(s[6])+s[3]*Ts,
    #                                    s[7],s[8]]
    #         deltay=s[5]-array_BSN[1]
    #         deltaz=s[6]-array_BSN[2]
    #         offsets=np.sqrt(deltay**2+deltaz**2)
    #         weights.append(np.array([self.getWeights(ri) for ri in offsets]))
            
    #     weights=np.array(weights).T  
    #     weighted_qtty=[]
        
    #     for i in range(0,len(sList[0].T)):
    #         tempBz=[]
    #         for data in sList:
    #             tempBz.append(data[indS][i])
    #         B=sum(np.array(tempBz)*weights[i])/sum(weights[i])
    #         weighted_qtty.append(B)
        
    #     shifted_time_series = np.asarray(s[0])      
    #     return weighted_qtty, shifted_time_series
    
    
    

    def alignInterp(self,datasets):
        """
        Align time series within datasets and interpolate data for each dataset at one-minute intervals.
    
        Parameters:
        datasets: Variable number of datasets, each containing time series and data arrays.
    
        Returns:
        list of numpy.ndarray: List of datasets with new data and time series.
        """
        
        # Extract time series and data arrays for each dataset
        time_series_list = [dataset[0] for dataset in datasets]
        #print(time_series_list)
        data_arrays_list = [dataset[1:] for dataset in datasets]
    
        # Find the common start and end times for all datasets
        common_start_time = max([min(time) for time in time_series_list])
        common_end_time = min([max(time) for time in time_series_list])
    
        # Create the common time series within the overlapping time range at one-minute intervals
        interpolated_time_series = np.arange(common_start_time, common_end_time, 60)
    
        # Interpolate data for each dataset
        interpolated_data = [
            [np.interp(interpolated_time_series, time_series, data) for data in data_arrays]
            for time_series, data_arrays in zip(time_series_list, data_arrays_list)
        ]
    
        # Create a list of datasets with new data and time series
        result_datasets = [
            [np.array(interpolated_time_series)] + [np.array(data) for data in data_arrays]
            for interpolated_time_series, data_arrays in zip(interpolated_time_series, interpolated_data)
        ]
        for dataset in result_datasets:
            dataset[0]=interpolated_time_series
        return result_datasets
    
    
    
    def WA_method(self,sList,indS):
        """
        USAGE: Would recommend calling function and iterating through indS values you want
        No functionality to pass in and iterate through a list of indices.
        Interpolation/propagation/weight determination are all vectorized so should be quick enough
    
        Parameters
        ----------
        sList : list/array
        List of spacecraft measurements. Elements in this list should be of the form:
        [t0,vx0,vy0,vz0,x0,y0,z0,DATA],[t1,vx1,vy1,vz1,x1,y1,z1,...].T lol.
        
        
        indS: int
        Index of data you want to average in s
        
        Returns
        -------
        sListNew: List
        New list of propagated/modified s/c data
        
        [sListNew[0][0],weightedBs]: List
        List containing averaged data and averaged data along with its
        interpolated time series
        
        
    
        """

        weights=[]
        i=0
        for s in sList:
            # Propagated location we are using here is (14Re,0,0)
            Ts=(10*self._Re-s[4])/(s[1])#calculates prop time
            s=[s[0]+Ts,s[1],s[2],s[3],np.array(s[4])+s[1]*Ts,
                                       np.array(s[5])+s[2]*Ts,
                                       np.array(s[6])+s[3]*Ts,
                                       s[7],s[8]] #modifies position and time in list
            sList[i]=s
            i+=1
        # I've commented out below to test without interpolation in time
        sListNew=self.alignInterp(sList)    
        #sListNew = sList
        i=0
        for s in sListNew:
            offsets=np.sqrt(s[5]**2+s[6]**2) #gets offsets and weights
            weights.append(np.array([self.getWeights(ri) for ri in offsets]))
            sListNew[i]=s
            i+=1
        weights=np.array(weights).T
    
        weightedBs=[]
        for i in range(0,len(np.array(sListNew[0]).T)):
            tempBz=[]
            for data in sListNew:
                tempBz.append(data[indS][i]) #weighted averages the quantity s[indS]
            B=sum(np.array(tempBz)*weights[i])/sum(weights[i])
            weightedBs.append(B)
        
        return sListNew[0][0],weightedBs
    
    
    def multiSC_WA_Propagate(self):
        
        new_time, new_E = self.WA_method(self._multiSC,8)
        new_time, new_P = self.WA_method(self._multiSC,7)
        
        return pd.DataFrame({'Time': new_time, 'Efield': new_E, 'Pressure': new_P})
    
    
    def multiSC_WA_Propagate_pData(self):
        
        new_time, new_Bz = self.WA_method(self._multiSC,7)
        new_time, new_n = self.WA_method(self._multiSC,8)
        new_time, new_v = self.WA_method(self._multiSC,9)
        
        return pd.DataFrame({'Time': new_time, 'Bz': new_Bz, 'n': new_n, 'v': new_v})
    
    
    def pairs_WA_propagate(self):
        
       ADpair = [self._ACE.to_numpy().T, self._DSCOVR.to_numpy().T]
       AWpair = [self._ACE.to_numpy().T, self._Wind.to_numpy().T]
       DWpair = [self._DSCOVR.to_numpy().T, self._Wind.to_numpy().T]
        
       new_time1, new_E1 = self.WA_method(ADpair,8)
       new_time1, new_P1 = self.WA_method(ADpair,7)
       
       new_time2, new_E2 = self.WA_method(AWpair,8)
       new_time2, new_P2 = self.WA_method(AWpair,7)
       
       new_time3, new_E3 = self.WA_method(DWpair,8)
       new_time3, new_P3 = self.WA_method(DWpair,7)
       
       return (pd.DataFrame({'Time': new_time1, 'Efield': new_E1, 'Pressure': new_P1}),
              pd.DataFrame({'Time': new_time2, 'Efield': new_E2, 'Pressure': new_P2}),
              pd.DataFrame({'Time': new_time3, 'Efield': new_E3, 'Pressure': new_P3}))
        
        