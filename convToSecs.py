# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:51:02 2023

@author: Ned
"""
import datetime

def seconds_since_start_of_year(day, hour, minute, second):
    # Create a datetime object for the given year and day of the year
    current_year = datetime.datetime.now().year
    start_of_year = datetime.datetime(current_year, 1, 1)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    second = (second)

    # Calculate the time difference between the given time and the start of the year
    time_delta = datetime.timedelta(days=day - 1, hours=hour, minutes=minute, seconds=second)

    # Calculate the total seconds
    total_seconds = time_delta.total_seconds()

    return total_seconds
