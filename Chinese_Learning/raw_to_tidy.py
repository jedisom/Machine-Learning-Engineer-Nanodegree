# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 06:56:12 2016
@author: jed.isom

This script was created to take the raw copy/paste data set and turn it into
a tidy dataset.  Mostly, it takes the punctuation and extra spacing out of the 
text and then calculates the seconds per character metric that will be used
as the Y, output, variable for supervised learning.
"""

#import libraries needed
import numpy as np
import datetime as dt

data_array={'names': ('date', 'time_spent', 'text_read'),
                      'formats': ('S1', 'f4', 'S1')}
#data_array = np.transpose([str, float, str])
raw_data = np.loadtxt("Raw_Chinese_Learning_Log.txt", dtype = data_array,
                      delimiter = "\t", skiprows = 1)
                      
