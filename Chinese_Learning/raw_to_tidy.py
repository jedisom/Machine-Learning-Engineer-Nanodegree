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
import pandas as pd
import datetime as dt
import os

#I got a HUGE amount of help importing Chinese characters from Excel to python 
#from these StackOverlow posts 
#http://stackoverflow.com/questions/29767898/how-to-read-excel-unicode-characters-using-python
#http://stackoverflow.com/questions/5879754/run-excel-file-from-python-error

#Setup port to Excel file where data is stored
import win32com.client
xl = win32com.client.gencache.EnsureDispatch('Excel.Application')
xl.Workbooks.Open(os.path.abspath(r'Raw_Chinese_Learning_Log.xlsx'))

#Initialize dataframe where raw data will be written
columns = ["date", "time_spent", "text_read"]
tidy_data = pd.DataFrame(columns = columns)

#Cycle through all rows with data and input data into tidy_data dataframe
row = 2
while xl.ActiveSheet.Cells(row, 1).Text != u'':
    print row    
    d = xl.ActiveSheet.Cells(row,1).Text
    t = xl.ActiveSheet.Cells(row,2).Text
    r = xl.ActiveSheet.Cells(row,3).Text
    tidy_data = tidy_data.append({'date': d, 'time_spent': t, 'text_read': r}, 
                                 ignore_index=True)
    row = row + 1                               

print tidy_data.head()
