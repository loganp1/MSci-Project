# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:36:38 2023

@author: Ned
"""

from datetime import datetime, timedelta
import pandas as pd
filename="wind_speed_1995-2023_1hour.csv"
df=pd.read_csv(filename)

def UnixConv(df, seconds=True, minutes=True,DayVar="Day",YearVar="Year"):
    if seconds==False:
        column_name = 'Seconds'
        df.insert(4, column_name, 0)
    if minutes==False:
        column_name = 'Minute'
        df.insert(3, column_name, 0)
    def convert_to_unix_timestamp(row):
        year = int(row[YearVar].item())
        day = int(row[DayVar].item())
        hour = int(row['Hour'].item())
        minute = int(row['Minute'].item())
        second = int(row['Seconds'].item())
        date = datetime(year, 1, 1) + timedelta(days=day - 1, hours=hour, minutes=minute, seconds=second)
        unix_timestamp = int((date - datetime(1970, 1, 1)).total_seconds())
        return unix_timestamp

    df['Time'] = df.apply(convert_to_unix_timestamp, axis=1)
    df.drop(columns=[YearVar, DayVar, 'Hour', 'Minute', 'Seconds'], inplace=True)
    df = df[['Time'] + [col for col in df.columns if col != 'Time']]
    return df
secs=input("Seconds present?, Y/N: ")
if secs=="N":
    secs=False
else:
    secs=True
    
minutes=input("Minutes present?, Y/N: ")
if minutes=="N":
    minutes=False
else:
    minutes=True

df=UnixConv(df,secs, minutes)
#df=UnixConv(df,secs)
df.to_csv(filename, index=False)

print("CSV save complete")