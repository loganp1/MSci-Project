# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:44:10 2023

@author: Ned
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dataCleaner(fname,zeros=0,skipcols=0):
    def invalidIdent(df,zeros,skipcols=0):
        for column in df.columns[skipcols:]:
            col=np.asarray(df[column])
            q1=np.percentile(col,25)
            q3=np.percentile(col,75)
            iqr=q3-q1
            for i in range(0,len(col)):
                if col[i]>q3+3*iqr or col[i]<q1-3*iqr:
                    temp=col[i]
                    zeros[i]=1
        plt.show()
        plt.hist(zeros)
        plt.show()
        print("Invalid data: "+str(100*round(np.sum(zeros)/len(zeros),5))+"%")
        return zeros
    df=pd.read_csv(fname)
    if zeros==0:
        zeros=np.zeros(len(df))
    zeros=invalidIdent(df,zeros)
    
    def remover(df, zeros):
        dfT=df.T
        for i in range(0,len(zeros)):
            if zeros[i]==1:
                dfT.pop(i)
        return dfT.T
    
    df=remover(df, zeros)
    df.to_csv(fname,index=False)
    return zeros






