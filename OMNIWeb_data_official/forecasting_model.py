# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:38:28 2024

@author: logan
"""


# Model from Burton et al. 1975

""" This file is designed to contain the forecasting function independently so that it can be imported and 
used in various other files necessary for this project."""

import numpy as np

# Create injection energy function of E
def F(E, d):
    
    if E < 0.5:
        F = 0
    else:
        F = d * (E - 0.5)
    return F


def SYM_forecast(SYM_i, P_i, P_iplus1, E_i):
    
    # Add constants

    dt = 60 # seconds in an minute, as one timestep is one minute for SYM/H

    # Unit gamma is used (1nT)
    gamma = 1 # BECAUSE OUR DATAFRAMES HAVE UNITS OF nT!!!

    # Set your parameter values (a, b, c)
    a = 3.6e-5
    b = 0.2 * gamma
    c = 20 * gamma
    d = -1e-3 * gamma
    
    derivative_term = b * (P_iplus1**0.5 - P_i**0.5)/dt
    
    return SYM_i + (-a * (SYM_i - b * np.sqrt(P_i) + c) + F(E_i, d) + derivative_term) * dt