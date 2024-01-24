# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:27:24 2023

@author: logan
"""

import numpy as np

def cross_correlation(x, y, interpolated_time_series):
    """
    Calculate the cross-correlation between two time series along with corresponding time lags.

    Parameters:
    - x: numpy array, time series data
    - y: numpy array, time series data
    - interpolated_time_series: numpy array, time series data used for time lags
    - max_lag: int, maximum lag for cross-correlation (default is None)

    Returns:
    - time_lags: numpy array, time lag series
    - cross_corr: numpy array, cross-correlation values
    """
    
    # Calculate the mean of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate x' and y' (subtract mean)
    x_prime = x - x_mean
    y_prime = y - y_mean

    # Calculate cross-correlation using the formula
    cross_corr = np.correlate(x_prime, y_prime, mode='full') / np.sqrt(np.sum(x_prime**2) * np.sum(y_prime**2))
    
    # Calculate the time lags corresponding to the cross-correlation values
    time_lags = np.arange(-len(x) + 1, len(x))

    # Calculate delta t for each time lag using the interpolated_time_series
    delta_t = interpolated_time_series[1] - interpolated_time_series[0]

    # Adjust time lags to correspond to the delta t in interpolated_time_series
    time_lags = time_lags * delta_t

    return time_lags, cross_corr


