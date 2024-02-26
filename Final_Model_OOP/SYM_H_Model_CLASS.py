# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:55:06 2024

@author: logan
"""

import numpy as np

class SYM_H_Model:
    
    def __init__(self, propagated_df, initial_value):
        
        self._df = propagated_df
        self._IV = initial_value
        
        
    # Injection energy function of E to go in model
    def F(self, E, d):
        
        if E < 0.5:
            F = 0
        else:
            F = d * (E - 0.5)
        return F


    def SYM_forecast_model(self,SYM_i, P_i, P_iplus1, E_i):
        
        '''Record Burton's final parameters so I don't forget when adjusting:
            
            a = 3.6e-5
            b = 0.2 * gamma
            c = 20 * gamma
            d = -1.5e-3 * gamma
            
            '''
            
        # Add constants

        dt = 60 # seconds in an minute, as one timestep is one minute for SYM/H

        # Unit gamma is used (1nT)
        gamma = 1 # BECAUSE OUR DATAFRAMES HAVE UNITS OF nT

        # Set your parameter values (a, b, c)
        a = 3.6e-5
        b = 0.2 * gamma
        c = 0 * gamma
        d = -1e-3 * gamma
        
        derivative_term = b * (P_iplus1**0.5 - P_i**0.5)/dt
        
        return SYM_i + (-a * (SYM_i - b * np.sqrt(P_i) + c) + self.F(E_i, d) + derivative_term) * dt
    
    
    def predict_SYM(self):
        
        sym_forecast = []
        sym_forecast.insert(0,self._IV)   # Add initial value we use to propagate through forecast
        current_sym = self._IV

        for i in range(len(self._df)-1):
            
            new_sym = self.SYM_forecast_model(current_sym,
                                             self._df['Pressure'][i],
                                             self._df['Pressure'][i+1],
                                             self._df['Efield'][i])
            sym_forecast.append(new_sym)
            current_sym = new_sym
            
        return sym_forecast
    
    
    
            

    
        