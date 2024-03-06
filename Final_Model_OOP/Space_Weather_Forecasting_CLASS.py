# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:34:21 2024

@author: logan
"""

import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Add path to the directory containing classes
sys.path.append(r"C:\Users\logan\OneDrive - Imperial College London\Uni\Year 4\MSci Project\MSci-Project"
                r"\Final_Model_OOP")

from SC_Propagation_CLASS import SC_Propagation
from SYM_H_Model_CLASS import SYM_H_Model
from align_and_interpolate import align_and_interpolate_datasets
from cross_correlation import cross_correlation


# Create final class which inherits the other 2's attributes
class Space_Weather_Forecast(SYM_H_Model, SC_Propagation):
    
    ''' 
    
    This class serves as the final integrative framework, orchestrating the collaboration among other classes 
    to execute conclusive space weather forecasting computations and statistical analyses. 
    
    '''
    
    def __init__(self, SC_dict, SYM_real, OMNI_data=None):
        
        self._SC_dict = SC_dict
        self._SYMr = SYM_real
        self._OMNI = OMNI_data
        self._ace_dfs = None 
        self._dsc_dfs = None 
        self._wnd_dfs = None
        self._omni_dfs = None
        
        
    def unix_to_DateTime(self):
        
        for spacecraft, df in self._SC_dict.items():
            
            df['DateTime'] = pd.to_datetime(df['Time'], unit='s')
            
        self._SYMr['DateTime'] = pd.to_datetime(self._SYMr['Time'], unit='s')
        
        
    def GetSCdf(self):
        
        return self._SC_dict
            
            
    def GetSCdata(self, sc_name):
        
        return self._SC_dict[sc_name]
    
    def GetSYMdata(self):
        
        return self._SYMr
    
    def GetSCsubDFs(self):
        
        return [self._ace_dfs,self._dsc_dfs,self._wnd_dfs]
    
    
    def SplitSubDFs(self,index1,index2):
        
        self._ace_dfs = [self._ace_dfs[index1:index2]]
        self._dsc_dfs = [self._dsc_dfs[index1:index2]]
        self._wnd_dfs = [self._wnd_dfs[index1:index2]]
        
        #return self._ace_dfs[index], self._dsc_dfs[index], self._wnd_dfs[index]
    
    
    def check_nan_values(self, sc_name):
        """
        Check for NaN values in the 'n' column and return a list of DateTime values.

        Parameters:
        - sc_name (str): Name of the spacecraft to check.

        Returns:
        - nan_datetime_values (list): List of DateTime values where 'n' column has NaN values.
        """

        # Get the DataFrame for the specified spacecraft
        df = self._SC_dict[sc_name]

        # Find rows where 'n' column has NaN values
        nan_mask = df['n'].isna()
        nan_datetime_values = df.loc[nan_mask, 'DateTime'].tolist()

        return nan_datetime_values
            
            
    def filter_dates(self, lower_date, upper_date):
        """
        Filter the DataFrames in self._SC_dict to include only rows within the specified date range.

        Parameters:
        - lower_date (datetime): Lower bound of the date range.
        - upper_date (datetime): Upper bound of the date range.

        Returns:
        - filtered_data (dict): Dictionary with filtered DataFrames.
        """

        filtered_data = {}
        for spacecraft, df in self._SC_dict.items():
            mask = (df['DateTime'] >= lower_date) & (df['DateTime'] <= upper_date)
            df_filtered = df[mask].copy()
            filtered_data[spacecraft] = df_filtered

        self._SC_dict = filtered_data    
        
        mask2 = (self._SYMr['DateTime'] >= lower_date) & (self._SYMr['DateTime'] <= upper_date)
        df_filtered2 = self._SYMr[mask2].copy()

        self._SYMr = df_filtered2
        
        
        
    # Create function which uses all the methods defined to perform the forecast
    def Forecast_SYM_H(self, sym0, class1, chosen_method, chosen_spacecraft = None):
        
        '''class1 is the propagation class (I've had to edit this as was calling required_form too often)'''
        
        # Ensure chosen_method is either 'single' or 'multi'
        assert chosen_method in ['single', 'multi', 'both', 'pair_combs'], \
        "Invalid value for chosen_method. Choose 'single', 'multi', 'both' or 'pair_combs'."
        
        # Extract initial SYM/H value ready for forecasting
        #sym0 = self._SYMr['SYM/H, nT'].values[0] ### OH MY GOD DONT USE THIS FOR SPLIT DATA 
        #print('ALERT: YOU ARE USING SAME SYM0 FOR ALL PERIODS - STOP IF DOING SPLIT PERIODS')
        # actually don't need this just input single sym0 value into fn if doing whole dataset not split
        
        # # Create an object from SC_Propagation sub-class to propagate the data downstream
        # class1 = SC_Propagation(self._SC_dict)

        # # Put the data in this class in required form for propagation (probably should do this auto in class)
        # # This function gives me so much grief!! It's poorly structured, but I'll add in this flag to track it so
        # # that it's only uses once, the first time it's needed.
        # if not self._class1_initialized:
        #     # If required_form() hasn't been called, call it
        #     class1.required_form()
        #     self._class1_initialized = True

        # If chosen_method = single: propagate chosen_spacecraft using single spacecraft method
        if chosen_method == 'single':
            assert chosen_spacecraft is not None, \
            "You must specify a spacecraft_name when chosen_method is 'single'."
            if chosen_spacecraft == 'ACE':
                df_prop = class1.singleSC_Propagate('ACE')
            if chosen_spacecraft == 'DSCOVR':
                df_prop = class1.singleSC_Propagate('DSCOVR')
            if chosen_spacecraft == 'Wind':
                df_prop = class1.singleSC_Propagate('Wind')
            if chosen_spacecraft == 'all':
                df_prop1 = class1.singleSC_Propagate('ACE')
                df_prop2 = class1.singleSC_Propagate('DSCOVR')
                df_prop3 = class1.singleSC_Propagate('Wind') 

        # If chosen_method == multi: propagate combined spacecraft data using WA method
        if chosen_method == 'multi':
            #class1.required_form()
            df_prop = class1.multiSC_WA_Propagate()

        # Now test 2nd class - SYM/H forecasting
        if chosen_spacecraft == 'all':
            class2a = SYM_H_Model(df_prop1,sym0)
            class2d = SYM_H_Model(df_prop2,sym0)
            class2w = SYM_H_Model(df_prop3,sym0)
            sym_forecast1 = class2a.predict_SYM()
            sym_forecast2 = class2d.predict_SYM()
            sym_forecast3 = class2w.predict_SYM()
            
            time_series1 = df_prop1['Time']
            time_series2 = df_prop2['Time']
            time_series3 = df_prop3['Time']
            
            return time_series1, time_series2, time_series3, sym_forecast1, sym_forecast2, sym_forecast3
        
        if chosen_method == 'both':
            
            # Comment out below line if splitting dataframe
            class1.required_form()
            
            df_prop1 = class1.singleSC_Propagate('ACE')
            df_prop2 = class1.singleSC_Propagate('DSCOVR')
            df_prop3 = class1.singleSC_Propagate('Wind')
            df_propm = class1.multiSC_WA_Propagate()
            
            class2a = SYM_H_Model(df_prop1,sym0)
            class2d = SYM_H_Model(df_prop2,sym0)
            class2w = SYM_H_Model(df_prop3,sym0)
            class2m = SYM_H_Model(df_propm,sym0)
            sym_forecast1 = class2a.predict_SYM()
            sym_forecast2 = class2d.predict_SYM()
            sym_forecast3 = class2w.predict_SYM()
            sym_forecastm = class2m.predict_SYM()
            
            time_series1 = df_prop1['Time']
            time_series2 = df_prop2['Time']
            time_series3 = df_prop3['Time']
            time_seriesm = df_propm['Time']
            #time_series1 = pd.to_datetime(df_prop1['Time'],unit='s')
            #time_series2 = pd.to_datetime(df_prop2['Time'],unit='s')
            #time_series3 = pd.to_datetime(df_prop3['Time'],unit='s')
            #time_seriesm = pd.to_datetime(df_propm['Time'],unit='s')
            
            return (time_seriesm, time_series1, time_series2, time_series3, sym_forecastm, sym_forecast1, 
                   sym_forecast2, sym_forecast3)
        
        if chosen_method == 'pair_combs':
            
            df_propAD, df_propAW, df_propDW = class1.pairs_WA_propagate()
            
            classAD = SYM_H_Model(df_propAD,sym0)
            classAW = SYM_H_Model(df_propAW,sym0)
            classDW = SYM_H_Model(df_propDW,sym0)
            sym_forecast1 = classAD.predict_SYM()
            sym_forecast2 = classAW.predict_SYM()
            sym_forecast3 = classDW.predict_SYM()
            
            time_series1 = df_propAD['Time']
            time_series2 = df_propAW['Time']
            time_series3 = df_propDW['Time']
            
            return (time_series1, time_series2, time_series3, 
                    sym_forecast1, sym_forecast2, sym_forecast3)
            
            
        else:
            class2 = SYM_H_Model(df_prop,sym0)
            sym_forecast = class2.predict_SYM()    
        
            time_series = df_prop['Time']
            
            return sym_forecast, time_series
        
    
    def Compare_Forecasts(self, chosen_method,chosen_spacecraft=None):

        if chosen_method == 'both':        

            #sym_forecast_mul, tm = self.Forecast_SYM_H('multi')
            tm, t1, t2, t3, sym_forecastm, sym_forecast1, sym_forecast2, sym_forecast3 = self.Forecast_SYM_H('both')
            return tm, t1, t2, t3, sym_forecast1, sym_forecast2, sym_forecast3, sym_forecastm
        
        
        
        elif chosen_method == 'single' and chosen_spacecraft == None:
            
            time_series, sym_forecast1, sym_forecast2, sym_forecast3 = self.Forecast_SYM_H('single','all')
            return time_series, sym_forecast1, sym_forecast2, sym_forecast3
        
        elif chosen_method == 'single':
            
            sym_forecast, time_series = self.Forecast_SYM_H(chosen_method,chosen_spacecraft)
            return time_series, sym_forecast

        
        
    def GetStats(self, chosen_pair):
        
        '''
        Function to calculate cross-correlation between a chosen pair of distributions
        chosen_ pair: ['spacecraft1/multi/realsym','spacecraft2/multi/realsym']
        chosen_stats: 'sep' or 'transverse sep' (I've removed this for now)
        '''

        tm, t1, t2, t3, sym1, sym2, sym3, sym_mul = self.Compare_Forecasts('both')

        sym_real = self.GetSYMdata()['SYM/H, nT']
        treal = self.GetSYMdata()['Time']
        
        # Turn series' into lists so we can index properly
        tm, t1, t2, t3, treal, sym_real = (tm.tolist(), t1.tolist(), t2.tolist(), t3.tolist(), treal.tolist(), 
                                           sym_real.tolist())
        
        # Form 2d lists for time series' + data
        ace_list = [t1,sym1]
        dsc_list = [t2,sym2]
        win_list = [t3,sym3]
        mul_list = [tm,sym_mul]
        real_list = [treal,sym_real]
        
        # Make dictionary so we can do everything in one go (instead of if statements for inputted pair)
        newdic = {'ACE':ace_list,'DSCOVR':dsc_list,'Wind':win_list,'multi':mul_list,'realsym':real_list}
        
        # Now apply which pair we've chosen - create common time series & corresponding data
        common_time, symA, symB = align_and_interpolate_datasets(newdic[chosen_pair[0]],
                                                                 newdic[chosen_pair[1]],len(sym_real))
        symA, symB = symA[0], symB[0]
        
        # Cross correlation
        time_delays,cross_corr_values = cross_correlation(symA, symB, common_time)
        peak_index = np.argmax(np.abs(cross_corr_values))

            

    def SplitTimes(self, split_times, keep_primary_data=False):
        
        # Set keep_primary_data to keep v, Bz and n data and propagate this instead of E & P
        # (You can obviously simply calculate E & P after propagation if necessary)
        
        ace_dfs = []
        dsc_dfs = []
        wnd_dfs = []
        # omni_dfs = []
        
        # Try and get all dfs in required form before splitting
        class_prop = SC_Propagation(self._SC_dict)
        class_prop.required_form(keep_primary_data=keep_primary_data)
        self._SC_dict = class_prop._SCdict
        
        # Split the DataFrame based on the min_max_times
        for i in range(len(split_times)):
            start_time, end_time = split_times[i]
            subset_df_ACE = self._SC_dict['ACE'][(self._SC_dict['ACE']['Time'] >= start_time) & 
                                                 (self._SC_dict['ACE']['Time'] <= end_time)].copy()
            ace_dfs.append(subset_df_ACE)
            subset_df_DSC = self._SC_dict['DSCOVR'][(self._SC_dict['DSCOVR']['Time'] >= start_time) &
                                                 (self._SC_dict['DSCOVR']['Time'] <= end_time)].copy()
            dsc_dfs.append(subset_df_DSC)
            subset_df_Wind = self._SC_dict['Wind'][(self._SC_dict['Wind']['Time'] >= start_time) &
                                                   (self._SC_dict['Wind']['Time'] <= end_time)].copy()
            wnd_dfs.append(subset_df_Wind)
            
            # Actually need OMNI split at propagated times so create another function to do that later after prop.
            # subset_df_omni = self._OMNI[(self._OMNI['Time'] >= start_time) &
            #                                        (self._OMNI['Time'] <= end_time)].copy()
            # omni_dfs.append(subset_df_omni)
            
        # Set the class attributes
        self._ace_dfs = ace_dfs
        self._dsc_dfs = dsc_dfs
        self._wnd_dfs = wnd_dfs
        #self._omni_dfs = omni_dfs
        
        
    def SplitOMNI(self,split_time_omni):
        
        # Don't need loop as will split separately in loop inside GetCC
            
        start_time, end_time = split_time_omni
    
        subset_df_omni = self._OMNI[(self._OMNI['Time'] >= start_time) &
                                    (self._OMNI['Time'] <= end_time)].copy()
        return subset_df_omni
    
    
    def GetSYM0(self,method,i):
         
        ##### CAREFUL NOT TO CHANGE THE CLASS WHEN DOING THIS JUST GET THE PROPAGATED TIME
        sym0 = 0 # doesn't matter, dummy variable as we'll just extract propagated times
        
        sc_dict = {'ACE': self._ace_dfs[i], 'DSCOVR': self._dsc_dfs[i], 'Wind': self._wnd_dfs[i]}
        myclass = Space_Weather_Forecast(SC_dict=sc_dict, SYM_real=self._SYMr)
        class_prop = SC_Propagation(sc_dict)
        
        if method == 'multi':
            sym1, t1 = myclass.Forecast_SYM_H(sym0, class_prop, 'multi')
            
        elif method == 'ACE':
            sym1, t1 = myclass.Forecast_SYM_H(sym0, class_prop, 'single', 'ACE')
            
        elif method == 'DSCOVR':
            sym1, t1 = myclass.Forecast_SYM_H(sym0, class_prop, 'single', 'DSCOVR')
            
        elif method == 'Wind':
            sym1, t1 = myclass.Forecast_SYM_H(sym0, class_prop, 'single', 'Wind')
        
        # Locate start time of method
        time0 = t1[0]
        
        # Time series don't match exactly so get closest time for initial sym value
        target_time = time0
        closest_time_index = (self._SYMr['Time'] - target_time).abs().idxmin()
        
        # Now you have the index of the row with the closest time value
        # You can use this index to get the corresponding 'SYM/H, nT' value
        initial_sym_val = self._SYMr.loc[closest_time_index, 'SYM/H, nT']
        
        return initial_sym_val
                                      
    
    def GetCC(self,chosen_pair):
        
        '''
        chosen_pair should be a list of 2 strings, two of: 
        'ACE', 'DSCOVR', 'Wind', 'multi', 'real'
        Preferably put 'real' second in list as len(element 1) is used which will make code run faster if not sym
        
        If doing multi vs OMNI: VERY IMPORTANT: MAKE SURE 'multi' IS FIRST ARGUMENT AS WE NEED TO GET ITS 
        PROPAGATED TIMES SO WE KNOW HOW TO SPLIT UP THE OMNI DATA TO MATCH IT.
        
        '''
            
        
        zvCCs = []
        maxCCs = []
        deltaTs = []
        
        # For pairs as well 
        zvCCsAD = []
        maxCCsAD = []
        deltaTsAD = []
        zvCCsAW = []
        maxCCsAW = []
        deltaTsAW = []
        zvCCsDW = []
        maxCCsDW = []
        deltaTsDW = []
        
        # I also want to return some of the datasets used to derive results for more insight
        
        # Want propagated: Bz, n, v which are used to derive E & P (we can always use them to plot E & P)
        
        
        
        for i in range(len(self._ace_dfs)):
            
            #start_time = self._ace_dfs[i]['Time'][0]
            
            sc_dict = {'ACE': self._ace_dfs[i], 'DSCOVR': self._dsc_dfs[i], 'Wind': self._wnd_dfs[i]}
            
            # We plug in the full sym as don't want to remove important outside-edge data 
            myclass = Space_Weather_Forecast(SC_dict=sc_dict, SYM_real=self._SYMr)
            
            
            
            # # Put sub-dataframe section in required form HERE
            
            # # Create an object from SC_Propagation sub-class to propagate the data downstream
            class_prop = SC_Propagation(sc_dict)

            # if first == True:
            #     # Put the data in this class in required form for propagation if 1st run otherwise will 
            #     # multiply by Re too many times as stored in the class ready for next run
            #     class_prop.required_form()
            #     # Put this class1 into the 1st argument for Forecast_SYM_H
        
            ### ERRRORRRRRRRRRRR ERRRORRRRRRRRR DONT FORGET TO ADD SYM0 METHOD FOR SINGLE SCs
            
            # For first in pair
            if chosen_pair[0] in ['ACE', 'DSCOVR', 'Wind']:
                pairs = False
                sym0 = self.GetSYM0(chosen_pair[0],i)
                sym1, t1 = myclass.Forecast_SYM_H(sym0, class_prop, 'single', chosen_pair[0])

            elif chosen_pair[0] == 'multi':
                pairs = False
                sym0 = self.GetSYM0('multi',i)
                sym1, t1 = myclass.Forecast_SYM_H(sym0, class_prop, 'multi')
            elif chosen_pair[0] == 'real':
                pairs = False
                sym1, t1 = myclass.GetSYMdata()['SYM/H, nT'], myclass.GetSYMdata()['Time']
                
            elif chosen_pair[0] == 'pair_combs':
                pairs = True
                sym0 = self.GetSYM0('multi',i) # using 3 SC multi to get sym0 for now, MAY NEED TO CHANGE
                tAD, tAW, tDW, symAD, symAW, symDW = myclass.Forecast_SYM_H(sym0, class_prop, 'pair_combs')
                
            # Now we can split the OMNI data if we need to based on the multi propagated times
            # Only do this inside elif statement below to optimise run time
            
            # We need to find the real sym at start of period - use propagated time from first in pair
            
                
            # For second in pair
            if chosen_pair[1] in ['ACE', 'DSCOVR', 'Wind']:
                sym0 = self.GetSYM0(chosen_pair[0],i)
                sym2, t2 = myclass.Forecast_SYM_H(class_prop, 'single', chosen_pair[1])
            elif chosen_pair[1] == 'multi':
                sym0 = self.GetSYM0('multi')
                sym2, t2 = myclass.Forecast_SYM_H(class_prop, 'multi')
            elif chosen_pair[1] == 'real':
                sym2, t2 = myclass.GetSYMdata()['SYM/H, nT'], myclass.GetSYMdata()['Time']
                
                ##### DONT USE CODE BELOW ITS NONSENSE!!!
                # # Code below is same as for OMNI data to get the times we need for real and make code faster
                # # Locate start and end time of multi (or could be other tbf, just chosen_pair[0])
                # time0 = t1.values[0]
                # timeX = t1.values[-1]
                
                # split_real = self.SplitOMNI([time0,timeX])
                # split_real = split_real.reset_index()
                
                # # Time series don't match exactly so get closest time for initial sym value
                # target_time = time0
                # closest_time_index = (self._SYMr['Time'] - target_time).abs().idxmin()
                
                # # Now you have the index of the row with the closest time value
                # # You can use this index to get the corresponding 'SYM/H, nT' value
                # initial_sym_val = self._SYMr.loc[closest_time_index, 'SYM/H, nT']
                
                # sym_class = SYM_H_Model(split_real,initial_sym_val)
                # sym_Real = sym_class.predict_SYM()
                # time_Real = split_real['Time']
                # sym2, t2 = sym_Real, time_Real
                
                
            elif chosen_pair[1] == 'OMNI':
                
                # Locate start and end time of multi (or could be other tbf, just chosen_pair[0])
                time0 = t1.values[0]
                timeX = t1.values[-1]
                
                split_OMNI = self.SplitOMNI([time0,timeX])
                split_OMNI = split_OMNI.reset_index()
                
                # Time series don't match exactly so get closest time for initial sym value
                target_time = time0
                closest_time_index = (self._SYMr['Time'] - target_time).abs().idxmin()
                
                # Now you have the index of the row with the closest time value
                # You can use this index to get the corresponding 'SYM/H, nT' value
                initial_sym_val = self._SYMr.loc[closest_time_index, 'SYM/H, nT']
                
                sym_class = SYM_H_Model(split_OMNI,initial_sym_val)
                sym_OMNI = sym_class.predict_SYM()
                time_OMNI = split_OMNI['Time']
                sym2, t2 = sym_OMNI, time_OMNI
                
                
            # Need an extra option for if doing OMNI vs real as OMNI needs to be first but need multi prop times
            # to get the OMNI times split
            
            if chosen_pair[0] == 'OMNI' and chosen_pair[1] == 'real':
                
                sym_ignore, t_ignore = myclass.Forecast_SYM_H(class_prop, 'multi')
                
                time0 = t_ignore.values[0]
                timeX = t_ignore.values[-1]
                
                split_OMNI = self.SplitOMNI([time0,timeX])
                split_OMNI = split_OMNI.reset_index()
                
                # Time series don't match exactly so get closest time for initial sym value
                target_time = time0
                closest_time_index = (self._SYMr['Time'] - target_time).abs().idxmin()
                
                # Now you have the index of the row with the closest time value
                # You can use this index to get the corresponding 'SYM/H, nT' value
                initial_sym_val = self._SYMr.loc[closest_time_index, 'SYM/H, nT']
                
                sym_class = SYM_H_Model(split_OMNI,initial_sym_val)
                sym_OMNI = sym_class.predict_SYM()
                time_OMNI = split_OMNI['Time']
                sym1, t1 = sym_OMNI, time_OMNI
                
                
            if pairs == True:
                paircount=0
                t2 = t2.tolist()
                for pair in [[tAD,symAD],[tAW,symAW],[tDW,symDW]]:
                    
                    t1 = pair[0].tolist()
                    
                    # Form 2d lists for time series' + data
                    list1 = [t1,pair[1]]
                    list2 = [t2,sym2]
                    
                    # Align and interpolate 2 desired sym datasets
                    common_time, sym1A, sym2A = align_and_interpolate_datasets(list1,list2,len(t1))
                    sym1A, sym2A = sym1A[0], sym2A[0]
                    
                    # Cross-correlation
                    time_delays,cross_corr_values = cross_correlation(sym1A, sym2A, common_time)
        
                    # Find the index where time_delays is 0
                    zero_delay_index = np.where(time_delays == 0)[0]
                    max_index = np.argmax(cross_corr_values)
                
                    
                    # Output max CC, zero value CC and time shift between these two values
                    maxCC = max(cross_corr_values)
                    zeroValCC = cross_corr_values[zero_delay_index][0]
                    deltaT = time_delays[max_index] - time_delays[zero_delay_index]
                    
                    if paircount == 0:
                        zvCCsAD.append(zeroValCC)
                        maxCCsAD.append(maxCC)
                        deltaTsAD.append(deltaT[0])
                    if paircount == 1:
                        zvCCsAW.append(zeroValCC)
                        maxCCsAW.append(maxCC)
                        deltaTsAW.append(deltaT[0])
                    if paircount == 2:
                        zvCCsDW.append(zeroValCC)
                        maxCCsDW.append(maxCC)
                        deltaTsDW.append(deltaT[0])
                    paircount+=1
            
                
            else:
                # Turn series' into lists so we can index properly
                t1, t2 = (t1.tolist(), t2.tolist())
                
                # Form 2d lists for time series' + data
                list1 = [t1,sym1]
                list2 = [t2,sym2]
                
                # Align and interpolate 2 desired sym datasets
                common_time, sym1A, sym2A = align_and_interpolate_datasets(list1,list2,len(t1))
                sym1A, sym2A = sym1A[0], sym2A[0]
                
                # Cross-correlation
                time_delays,cross_corr_values = cross_correlation(sym1A, sym2A, common_time)
    
                # Find the index where time_delays is 0
                zero_delay_index = np.where(time_delays == 0)[0]
                max_index = np.argmax(cross_corr_values)
            
                
                # Output max CC, zero value CC and time shift between these two values
                maxCC = max(cross_corr_values)
                zeroValCC = cross_corr_values[zero_delay_index][0]
                deltaT = time_delays[max_index] - time_delays[zero_delay_index]
                
                zvCCs.append(zeroValCC)
                maxCCs.append(maxCC)
                deltaTs.append(deltaT[0])
            
            print(i)
            #first = False
           
        # Get deltaTs as list of elements (not list of lots of 1-element lists!)
        
        if pairs == True:
            
            deltaTsAD = [arr for arr in deltaTsAD]  
            deltaTsAW = [arr for arr in deltaTsAW]   
            deltaTsDW = [arr for arr in deltaTsDW]   
               
            return zvCCsAD, maxCCsAD, deltaTsAD, zvCCsAW, maxCCsAW, deltaTsAW, zvCCsDW, maxCCsDW, deltaTsDW
        
        else:
                
            deltaTs = [arr for arr in deltaTs]    
               
            return zvCCs, maxCCs, deltaTs
                
    
        