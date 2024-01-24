# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:58:09 2023

@author: logan
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:21:41 2023

@author: Ned
"""

import numpy as np
import matplotlib.pyplot as plt
def MaxCrossCorr(t,x,y,legend = ["X", "Y"]):
    plt.plot(t,x)
    plt.plot(t,y)
    plt.xlabel("Time")
    plt.ylabel("$B_z$ nT")
    plt.legend(legend)
    plt.show()
    vals=[]
    for i in range(0,len(x)):    
        temp=np.corrcoef(x,y)[1,0]
        vals.append(temp)
        x=np.roll(x,1)
    shiftedX=np.roll(x,np.argmax(vals))
    plt.plot(t,shiftedX)
    plt.plot(t,y)
    plt.xlabel("Time")
    plt.ylabel("$B_z$ nT")
    legend[0]+= " Shifted"
    plt.legend(legend)
    plt.show()
    plt.plot(np.array(t),vals)
    plt.xlabel("Time After First Measurement, s")
    plt.ylabel("Correlation Coefficient")
    plt.show()
    return vals, t[np.argmax(vals)]

def find_closest_index(lst, target_value):
    """
    Find the index of the value in the list closest to the target value.

    Parameters:
    - lst: The list of values.
    - target_value: The value to which the closest value in the list will be found.

    Returns:
    - The index of the closest value in the list.
    """
    closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - target_value))
    return closest_index

def crossCorrShift(t,x,y,tshift,plot=False):
    indShift=find_closest_index(t, tshift)
    print(indShift)
    x=np.roll(x,-1*indShift)    
    zeroCorr=np.corrcoef(x,y)[1,0]
    vals=[]
    if plot==True:
        plt.plot(t,x)
        plt.plot(t,y)
        plt.show()
    
    for i in range(0,len(x)):    
        temp=np.corrcoef(x,y)[1,0]
        vals.append(temp)
        x=np.roll(x,1)
    vals=vals[::-1]
    vals=np.roll(vals,int(len(vals)/2))
    t=t-np.mean(t)
    if plot==True:
        plt.plot(t,vals)    
        plt.axvline(0, linestyle='dashed')
        plt.xlabel("$\Delta t$")
        plt.ylabel("Correlation Coefficient")
        plt.show()
    
    offSet=t[np.argmax(vals)]-t[find_closest_index(t, 0)]
    return vals, offSet,zeroCorr


    
