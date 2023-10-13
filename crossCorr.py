# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:21:41 2023

@author: Ned
"""

import numpy as np
import matplotlib.pyplot as plt
def crossCorr(t,x,y):
    plt.plot(t,x)
    plt.plot(t,y)
    plt.legend(["X", "Y"])
    plt.show()
    vals=[]
    for i in range(0,len(x)):    
        temp=np.corrcoef(x,y)[1,0]
        vals.append(temp)
        x=np.roll(x,1)
    shiftedX=np.roll(x,np.argmax(vals))
    plt.plot(t,shiftedX)
    plt.plot(t,y)
    plt.legend(["Shifted X", "Y"])
    plt.show()
    return vals, t[np.argmax(vals)]

    