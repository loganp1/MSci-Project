# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:55:25 2024

@author: logan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:05:14 2024

@author: Ned
"""

import numpy as np
#from datetime import datetime

"""
DSCOVR=pd.read_csv("testDSCOVR.csv").to_numpy().T
ACE=pd.read_csv("testACE.csv").to_numpy().T
WIND=pd.read_csv("testWIND.csv").to_numpy().T
ART=pd.read_csv("testARTEMIS.csv").to_numpy().T
sList=[DSCOVR,ACE,WIND]
"""
def getWeights(ri):
    r0=50*6378
    return np.exp(-1*ri/r0)

def weightedAv(sList,sRef,indS):
    """
    

    Parameters
    ----------
    sList : list/array
    List of spacecraft measurements. Elements in this list should be of the form:
    [t0,vx0,vy0,vz0,x0,y0,z0,DATA],[t1,vx1,vy1,vz1,x1,y1,z1,...].T lol.
        
    sRef: list/array                      
    Target: [[t0,x0,y0,z0,DATA],[t1,...].T lol].
    
    indS: int
    Index of data you want to average in s
    
    Returns
    -------
    weightedBs : TYPE
        DESCRIPTION.

    """
    weights=[]
    for s in sList:
        Ts=(sRef[1]-s[4])/(s[1])
        s=[s[0]+Ts,s[1],s[2],s[3],np.array(s[4])+s[1]*Ts,
                                   np.array(s[5])+s[2]*Ts,
                                   np.array(s[6])+s[3]*Ts,
                                   s[7],s[8]] # I removed an s[9] from here as only have E and P = 9 total rows
        deltay=s[5]-sRef[2]
        deltaz=s[6]-sRef[3]
        offsets=np.sqrt(deltay**2+deltaz**2)
        weights.append(np.array([getWeights(ri) for ri in offsets]))
        #datetime_objects = [datetime.utcfromtimestamp(int(ts)) for ts in s[0]]
        #plt.plot(datetime_objects,s[4])
    weights=np.array(weights).T  
    weightedBs=[]
    for i in range(0,len(sList[0].T)):
        tempBz=[]
        for data in sList:
            tempBz.append(data[indS][i])
        B=sum(np.array(tempBz)*weights[i])/sum(weights[i])
        weightedBs.append(B)
    #plt.show()   
    #plt.plot(weightedBs)
    #plt.plot(sRef[6])
    return weightedBs


    
   
            
            
    
        
"""      
#plt.plot([datetime.utcfromtimestamp(int(ts)) for ts in ART[0]],ART[1])
#plt.show()
        
            
test=weightedAv(sList,ART,9)
test2=MaxCrossCorr(ART[0], ART[6], test)
print(test2[1]/60)
"""     
    