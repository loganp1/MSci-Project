# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:15:05 2023

@author: Ned
"""

import pytplot
import pyspedas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytplot import tplot
from datetime import datetime, timedelta
def convert_to_unix_timestamp(date_strings):
    datetime_objects = [datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S') for date_string in date_strings]
    return [int(dt.timestamp()) for dt in datetime_objects]
def datetime_to_unix_seconds(datetimes):
    """
    Convert a list/array of datetimes to Unix seconds.

    Parameters:
    datetimes (list or array): List/array of datetime objects.

    Returns:
    list: List of Unix timestamps in seconds corresponding to the input datetimes.
    """
    return [int(dt.timestamp()) for dt in datetimes]
nameys=[['dscovr','wind','ace','themis','omni'],['data',['mag','orb','fc'],['mfi','swe','orbit'],['fgm','state']],
        [['SYM_D','SYM_H','ASY_D','ASY_H'],
         [['dsc_h0_mag_B1GSE','dsc_h0_mag_B1SDGSE'],['dsc_orbit_GSE_POS'],['dsc_h1_fc_V_GSE']],
         [['BGSE','BRMSGSE'],['U_eGSE','UceGSE'],['GSE_POS','GSE_VEL']],
         [['thc_fgs_gse','thc_fgl_gse','thc_fgh_gse'],['thc_pos_gse']]]]
for i in range(len(nameys[0])):
    print(f"{nameys[0][i]}, {i}")
def getData(i, trange): 
    data=[]
    if i==4:
        data=pyspedas.omni.data(trange,varnames=['SYM_D','SYM_H','ASY_D','ASY_H'], time_clip=True,notplot=True, datatype='1min')
    elif i==0:
        prick={'dsc_h0_mag_B1GSE':pyspedas.dscovr.mag(trange,datatype='h0',notplot=True)['dsc_h0_mag_B1GSE'],
               'dsc_h0_mag_B1SDGSE':pyspedas.dscovr.mag(trange,notplot=True)['dsc_h0_mag_B1SDGSE'] }
        data.append(prick)
        data.append(pyspedas.dscovr.fc(trange,notplot=True)['dsc_h1_fc_V_GSE'])
        data.append(pyspedas.dscovr.orb(trange,notplot=True)['dsc_orbit_GSE_POS'])        
        
    elif i==1:
        data.append(pyspedas.wind.mfi(trange,notplot=True,varnames=['BGSE','BRMSGSE'], datatype='h0',time_clip=True))
        data.append(pyspedas.wind.swe(trange,notplot=True,varnames=['U_eGSE'],datatype='h5',time_clip=True))
        data.append(pyspedas.wind.orbit(trange,notplot=True,varnames=['GSE_POS'],time_clip=True))
    elif i==3:
        data.append(pyspedas.themis.fgm(trange,notplot=True,varnames=['thc_fgs_gse'],level='l2'))
        data.append(pyspedas.themis.state(trange,notplot=True,varnames=['thc_pos_gse'],level='l1'))
    elif i==2:
        data.append(pyspedas.ace.mfi(trange,notplot=True,varnames=['BGSEc'], datatype='h0'))
        data.append(pyspedas.ace.swe(trange,notplot=True,varnames=['V_GSE','SC_pos_GSE'],datatype='h0'))
    return data
def string_to_datetime(date_string):
    return datetime.strptime(date_string, '%Y-%m-%d/%H:%M:%S')
#test=getData(3,['2018-11-5/13:04:06', '2018-11-5:15:04:06'])
#%%
#BArt=test[0]['dsc_h0_mag_B1GSE']['y']
#print(type(test[0]['BGSEc']['x'][0]))
from scipy.interpolate import interp1d





def multiInterp(target_time_series, convert_to_unix_epoch,*args):
    target_time_numeric = np.array([t.timestamp() for t in target_time_series])

    interpolated_data = []

    for time_series, data_series in args:
        if convert_to_unix_epoch:
            time_numeric = np.array([t.timestamp() for t in time_series])
        else:
            time_numeric = np.array(time_series)

        interp_function = interp1d(time_numeric, data_series, kind='linear', fill_value='extrapolate')
        interpolated_data_series = interp_function(target_time_numeric)

        interpolated_data.append(interpolated_data_series)

    

    interpolated_data = [
        [value for value in numeric_series] for numeric_series in interpolated_data
    ]
    interpolated_data.insert(0,datetime_to_unix_seconds(target_time_series))

    return interpolated_data



def dlFormat(trange,fname):
    headers=["Time","Bx","By","Bz","vx","vy","vz","x","y","z",]
    windDat=getData(1,trange) #initial data from wind
    time=trim_datetimes(trange, windDat[0]['BGSE']['x'])# one minute timestamps - our target sample times
    windDat=((windDat[0]['BGSE']['x'],windDat[0]['BGSE']['y'].T[0]),
             (windDat[0]['BGSE']['x'],windDat[0]['BGSE']['y'].T[1]),
             (windDat[0]['BGSE']['x'],windDat[0]['BGSE']['y'].T[2]),
             (windDat[1]['U_eGSE']['x'],windDat[1]['U_eGSE']['y'].T[0]),
             (windDat[1]['U_eGSE']['x'],windDat[1]['U_eGSE']['y'].T[1]),
             (windDat[1]['U_eGSE']['x'],windDat[1]['U_eGSE']['y'].T[2]),
             (windDat[2]['GSE_POS']['x'],windDat[2]['GSE_POS']['y'].T[0]),
             (windDat[2]['GSE_POS']['x'],windDat[2]['GSE_POS']['y'].T[1]),
             (windDat[2]['GSE_POS']['x'],windDat[2]['GSE_POS']['y'].T[2]))
    windDat=np.array(multiInterp(time,True,*windDat)).T
    df=pd.DataFrame(windDat).interpolate(method='linear',limit=10)
    df.to_csv(fname+"WIND.csv",header=headers, index=False)
    DSCDat=getData(0,trange)#getting dscovr data
    
    DSCDat=((DSCDat[0]['dsc_h0_mag_B1GSE']['x'],DSCDat[0]['dsc_h0_mag_B1GSE']['y'].T[0]),
            (DSCDat[0]['dsc_h0_mag_B1GSE']['x'],DSCDat[0]['dsc_h0_mag_B1GSE']['y'].T[1]),
            (DSCDat[0]['dsc_h0_mag_B1GSE']['x'],DSCDat[0]['dsc_h0_mag_B1GSE']['y'].T[2]),
            (DSCDat[1]['x'],DSCDat[1]['y'].T[0]),
            (DSCDat[1]['x'],DSCDat[1]['y'].T[1]),
            (DSCDat[1]['x'],DSCDat[1]['y'].T[2]),
            (DSCDat[2]['x'],DSCDat[2]['y'].T[0]),
            (DSCDat[2]['x'],DSCDat[2]['y'].T[1]),
            (DSCDat[2]['x'],DSCDat[2]['y'].T[2]))
    DSCDat=np.array(multiInterp(time,True, *DSCDat)).T
    df=pd.DataFrame(DSCDat).interpolate(method='linear',limit=10)
    df.to_csv(fname+"DSCOVR.csv",header=headers, index=False)
    
    aceDat=getData(2,trange)
    aceDat=((aceDat[0]['BGSEc']['x'],aceDat[0]['BGSEc']['y'].T[0]),
            (aceDat[0]['BGSEc']['x'],aceDat[0]['BGSEc']['y'].T[1]),
            (aceDat[0]['BGSEc']['x'],aceDat[0]['BGSEc']['y'].T[2]),
            (aceDat[1]['V_GSE']['x'],aceDat[1]['V_GSE']['y'].T[0]),
            (aceDat[1]['V_GSE']['x'],aceDat[1]['V_GSE']['y'].T[1]),
            (aceDat[1]['V_GSE']['x'],aceDat[1]['V_GSE']['y'].T[2]),
            (aceDat[1]['SC_pos_GSE']['x'],aceDat[1]['SC_pos_GSE']['y'].T[0]),
            (aceDat[1]['SC_pos_GSE']['x'],aceDat[1]['SC_pos_GSE']['y'].T[1]),
            (aceDat[1]['SC_pos_GSE']['x'],aceDat[1]['SC_pos_GSE']['y'].T[2]))
    aceDat=np.array(multiInterp(time,True,*aceDat)).T
    df=pd.DataFrame(aceDat).interpolate(method='linear',limit=10)
    df.to_csv(fname+"ACE.csv",header=headers, index=False)
    
    themData=getData(3,trange)
    themData=((themData[0]['thc_fgs_gse']['x'],themData[0]['thc_fgs_gse']['y'].T[0]),
              (themData[0]['thc_fgs_gse']['x'],themData[0]['thc_fgs_gse']['y'].T[1]),
              (themData[0]['thc_fgs_gse']['x'],themData[0]['thc_fgs_gse']['y'].T[2]),
              (themData[1]['thc_pos_gse']['x'],themData[1]['thc_pos_gse']['y'].T[0]),
              (themData[1]['thc_pos_gse']['x'],themData[1]['thc_pos_gse']['y'].T[1]),
              (themData[1]['thc_pos_gse']['x'],themData[1]['thc_pos_gse']['y'].T[2]),)
    themData=np.array(multiInterp(time,False,*themData)).T
    df=pd.DataFrame(themData).interpolate(method='linear',limit=10)
    headers=['Time','Bx','By','Bz','x','y','z']
    df.to_csv(fname+"ARTEMIS.csv",header=headers, index=False)
    
    return windDat, DSCDat, aceDat, themData
def trim_datetimes(time_range, datetime_list):
    start_time_str, end_time_str = time_range
    
    start_datetime = datetime.strptime(start_time_str, '%Y-%m-%d/%H:%M:%S')
    end_datetime = datetime.strptime(end_time_str, '%Y-%m-%d/%H:%M:%S')
    
    trimmed_datetimes = [dt for dt in datetime_list if start_datetime <= dt <= end_datetime]
    
    return trimmed_datetimes
temp=dlFormat(['2018-11-5/13:04:06', '2018-11-5/15:04:06'],"test")

#%%
import time
import random
t1=time.time()
from timefuncs import periodToTrange
periods=np.loadtxt("modPeriods.txt")
for i in range(5):
    j=random.randint(0,len(periods))
    print(j)
    trange=periodToTrange(periods[j])
    dlFormat(trange,str(i))
print(time.time()-t1)






