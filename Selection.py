# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:57:09 2023

@author: Ned
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("ARTEMIS16to19.csv")
cols=df.columns.values.tolist()
Xsse=cols[6]
Xgse=cols[5]
valid=0
periods=[[]]
valid=0
for i in range(0,len(df["Seconds"])):
    valTemp=valid
    temp1=df[Xsse][i]
    temp2=df[Xgse][i]
    if df[Xsse][i]> 0 and df[Xgse][i]> 14.11:
        valid = 1
    else:
        valid=0
    
    if valid==1 and valTemp==0:
        time=[df["YEAR"][i],df["DOY"][i],df["Hour"][i],df["Minute"][i],df["Seconds"][i]]
        periods.append([])
        periods[len(periods)-1].append(time)
    if valid==0 and valTemp==1:
        time=[df["YEAR"][i],df["DOY"][i],df["Hour"][i],df["Minute"][i],df["Seconds"][i]]
        periods[len(periods)-1].append(time)


   
periods.pop(0)

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

distances=[]
for i in range(0,len(periods)):
    distances.append(seconds_since_start_of_year(*periods[i][1][1:])-seconds_since_start_of_year(*periods[i][0][1:]))
      
    
plt.hist(np.array(distances)/(60*60*24))
plt.xlabel("Days")
plt.ylabel("Frequency")
print(periods[np.argmax(distances)])
print(max(distances)/(60*60*24))
plt.show()




        
