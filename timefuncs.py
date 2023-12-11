# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:08:48 2023

@author: Ned
"""

import pytplot
import pyspedas
from datetime import datetime

def unix_time_to_date(unix_time):
    dt_object = datetime.utcfromtimestamp(unix_time)
    formatted_date = dt_object.strftime('%Y-%m-%d/%H:%M:%S')
    return formatted_date
def periodToTrange(period):
    t1=unix_time_to_date(period[0])
    t2=unix_time_to_date(period[1])
    return [t1,t2]

def datetime_to_unix_seconds(datetimes):
    """
    Convert a list/array of datetimes to Unix seconds.

    Parameters:
    datetimes (list or array): List/array of datetime objects.

    Returns:
    list: List of Unix timestamps in seconds corresponding to the input datetimes.
    """
    return [int(dt.timestamp()) for dt in datetimes]


def unix_seconds_to_datetime(unix_seconds):
    """
    Convert Unix seconds to a list/array of datetimes.

    Parameters:
    unix_seconds (list or array): List/array of Unix timestamps in seconds.

    Returns:
    list: List of datetime objects corresponding to the input Unix timestamps.
    """
    return [datetime.fromtimestamp(ts) for ts in unix_seconds]


def convert_to_datetime(date_strings):
    return [datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S') for date_string in date_strings]



