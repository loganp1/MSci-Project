# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:08:37 2023

@author: Ned
"""
import numpy as np

def align_and_interpolate_datasets(dataset1, dataset2, num_points):
    """
    Align time series within two datasets and interpolate data for each dataset to the specified number of points.

    Parameters:
    dataset1 (list): List containing time series and data arrays for multiple datasets.
    dataset2 (list): List containing time series and data arrays for multiple datasets.
    num_points (int): The desired number of points for the interpolated time series.

    Returns:
    interpolated_time_series (numpy.ndarray): The interpolated time series for both datasets.
    interpolated_data1 (list of numpy.ndarray): Interpolated data arrays for each dataset in dataset1.
    interpolated_data2 (list of numpy.ndarray): Interpolated data arrays for each dataset in dataset2.
    """
    time_series1 = dataset1[0]
    data_arrays1 = dataset1[1:]

    time_series2 = dataset2[0]
    data_arrays2 = dataset2[1:]

    # Find the common start and end times for both datasets
    common_start_time = max(time_series1[0], time_series2[0])
    common_end_time = min(time_series1[-1], time_series2[-1])

    # Create the common time series within the overlapping time range
    interpolated_time_series = np.linspace(common_start_time, common_end_time, num_points)

    # Interpolate data for each dataset in dataset1
    interpolated_data1 = [np.interp(interpolated_time_series, time_series1, data) for data in data_arrays1]

    # Interpolate data for each dataset in dataset2
    interpolated_data2 = [np.interp(interpolated_time_series, time_series2, data) for data in data_arrays2]

    return interpolated_time_series, np.array(interpolated_data1), np.array(interpolated_data2)