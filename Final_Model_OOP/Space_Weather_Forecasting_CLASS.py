# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:34:21 2024

@author: logan
"""

import matplotlib.pyplot as plt
from SC_Propagation_CLASS import SC_Propagation
from SYM_H_Model_CLASS import SYM_H_Model


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
        
        
    # Create function which uses all the methods defined to perform the forecast
    def Forecast_SYM_H(self, chosen_method, chosen_spacecraft = None):
        
        # Ensure chosen_method is either 'single' or 'multi'
        assert chosen_method in ['single', 'multi'], \
        "Invalid value for chosen_method. Choose 'single' or 'multi'."
        
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

        # If chosen_method == multi: propagate combined spacecraft data using WA method
        if chosen_method == 'multi':
            df_prop = class1.multiSC_WA_Propagate()

        # Now test 2nd class - SYM/H forecasting

        class2 = SYM_H_Model(df_prop,sym0)
        sym_forecast = class2.predict_SYM()    
        
        time_series = df_prop['Time']
        
        return sym_forecast, time_series
        
        