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
Xsse=cols[2] #this assumes SSE x co-ord is 3rd column
Xgse=cols[1] #this assumes GSE x co-ord is 2nd column
valid=0
periods=[[]]
valid=0
for i in range(0,len(df["Time"])):
    valTemp=valid
    if df[Xsse][i]> 0 and df[Xgse][i]> 14.11:
        valid = 1
    else:
        valid=0
    
    if valid==1 and valTemp==0:
        time=df["Time"][i]
        periods.append([])
        periods[len(periods)-1].append(time)
    if valid==0 and valTemp==1:
        time=df["Time"][i]
        periods[len(periods)-1].append(time)


   
periods.pop(0)



distances=[]
for i in range(0,len(periods)-1):
    distances.append(periods[i][1]-periods[i][0])

plt.hist(np.array(distances)/(60*60*24))
plt.xlabel("Days")
plt.ylabel("Frequency")
print(periods[np.argmax(distances)])
print(max(distances)/(60*60*24))
plt.show()
distanceDays=np.array(distances)/(86400)





        
