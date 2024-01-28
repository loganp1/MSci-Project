# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 19:26:48 2024

@author: logan
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:41:51 2024

@author: Ned
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from crossCorr import crossCorrShift
def singleSpacecraft(s,sRef,indS,indRef,plotVar=False):
    """
    s : list/array
    Spacecraft data: [[t0,vx0,vy0,vz0,x0,y0,z0,DATA],[t1,vx1,vy1,vz1,x1,y1,z1,...].T lol.
                            
    sRef: list/array                      
    Target: [[t0,x0,y0,z0,DATA],[t1,...].T lol]
    
    indS: int
    Index of data you want to crosscorrelate in s
    
    indRef: int
    Index of data you want to crosscorrelate in sRef
    
    plotVar:Boolean
    Normally false - if true will plot
    -------
    s: Array.
    Propagated spacecraft data
    """
    def propagate(s, sRef):
        s=np.array(s)
        sRef=np.array(sRef)
        vxAv=np.mean(s[1])
        vyAv=np.mean(s[2])
        vzAv=np.mean(s[3])
        Ts=(sRef[1]-s[4])/(s[1])
        s=[s[0],s[1],s[2],s[3],np.array(s[4])+s[1]*Ts,
                                  np.array(s[5])+s[2]*Ts,
                                  np.array(s[6])+s[3]*Ts]
        return s, Ts
    
    
    
    # def analyse(s,sRef,Ts,plotVar=False):
    #     crossCorrelation=crossCorrShift(sRef[0], sRef[indRef], s[indS], np.mean(Ts),plot=plotVar)
    #     return crossCorrelation
    # temp=propagate(s,sRef)
    # temp2=analyse(s,sRef,temp[1],plotVar)
    # return temp2

"""
DSCOVR=pd.read_csv("testDSCOVR.csv").to_numpy().T
ACE=pd.read_csv("testACE.csv").to_numpy().T
WIND=pd.read_csv("testWIND.csv").to_numpy().T
ART=pd.read_csv("testARTEMIS.csv").to_numpy().T

test=singleSpacecraft(DSCOVR, ART,9,6,True)
"""

def propagate(s, sRef):
    s=np.array(s)
    sRef=np.array(sRef)
    vxAv=np.mean(s[1])
    vyAv=np.mean(s[2])
    vzAv=np.mean(s[3])
    Ts=(sRef[1]-s[4])/(s[1])
    s=[s[0],s[1],s[2],s[3],np.array(s[4])+s[1]*Ts,
                              np.array(s[5])+s[2]*Ts,
                              np.array(s[6])+s[3]*Ts]
    
    return s, Ts