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
    
    def __init__(self, SC_dict, SYM_real, df_OMNI_BSN=None):
        
        self._SC_dict = SC_dict
        self._SYMr = SYM_real
        self._OMNI = df_OMNI_BSN
        
        
    def unix_to_DateTime(self):
        
        for spacecraft, df in self._SC_dict.items():
            
            df['DateTime'] = pd.to_datetime(df['Time'], unit='s')
            
        self._SYMr['DateTime'] = pd.to_datetime(self._SYMr['Time'], unit='s')
            
            
    def GetSCdata(self, sc_name):
        
        return self._SC_dict[sc_name]
    
    def GetSYMdata(self):
        
        return self._SYMr
    
    
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
    def Forecast_SYM_H(self, chosen_method, chosen_spacecraft = None):
        
        # Ensure chosen_method is either 'single' or 'multi'
        assert chosen_method in ['single', 'multi', 'both'], \
        "Invalid value for chosen_method. Choose 'single', 'multi' or 'both'."
        
        # Extract initial SYM/H value ready for forecasting
        sym0 = self._SYMr['SYM/H, nT'].values[0]

        # Create an object from SC_Propagattion sub-class to propagate the data downstream
        class1 = SC_Propagation(self._SC_dict)

        # Put the data in this class in required form for propagation (probably should do this auto in class)
        class1.required_form()

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

        
        
    def GetStats(self, chosen_pair, chosen_stats):
        
        '''
        Function to calculate cross-correlation between a chosen pair of distributions
        chosen_ pair: ['spacecraft1/multi/realsym','spacecraft2/multi/realsym']
        chosen_stats: 'sep' or 'transverse sep'
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
                                                                 chosen_pair[1],len(sym_real))
        symA, symB = symA[0], symB[0]
        
        # Cross correlation
        time_delays,cross_corr_values = cross_correlation(symA, symB, common_time)
        peak_index = np.argmax(np.abs(cross_corr_values))

                                                  
    
    
        