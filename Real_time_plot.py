# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 14:18:29 2021

@author: Chaman Gupta
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt


files = glob.glob(r'E:\Shared drives\Pauzauskie Team Drive\ASG\NiND (1)\Laser Heating\02232021\LHR2**\*.txt',recursive = True) 


for file in files:
    df1=pd.read_csv(file,header=1,sep=',')
    print(df1)
    #fig = plt.figure()
    #plt.subplot(2, 1, 1)
    #plt.plot(df1.iloc[:,[1]],df1.iloc[:,[2]])
    
    #plt.subplot(2, 1, 2)
    #plt.plot(df1.iloc[:,[3]],df1.iloc[:,[4]])
    #plt.show() # this wil stop the loop until you close the plot