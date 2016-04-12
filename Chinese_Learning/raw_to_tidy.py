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
    #print row    
    date_text = xl.ActiveSheet.Cells(row,1).Text
    date_int = map(int, date_text.split("/"))
    d = dt.date(date_int[0], date_int[1], date_int[2])
    #d = xl.ActiveSheet.Cells(row,1).NumberFormat = "yyyy, mm, dd"
    t = int(xl.ActiveSheet.Cells(row,2).Text)
    r = xl.ActiveSheet.Cells(row,3).Text
    tidy_data = tidy_data.append({'date': d, 'time_spent': t, 'text_read': r}, 
                                 ignore_index=True)
    row = row + 1                               

print tidy_data.head()
test = tidy_data.head()

###
#Clean up text_read column
###

import string
import sys
import unicodedata
#(1) Remove numbers and whitespace
#http://stackoverflow.com/questions/12851791/removing-numbers-from-string
#http://stackoverflow.com/questions/11066400/remove-punctuation-from-unicode-formatted-strings

exclude = unicode(string.digits) + '\n' + ' '
#test['text_read'][0].translate({ord(n): None for n in exclude})
test2 = test['text_read'].map(lambda x: x.translate({ord(n): None for n in exclude}))

#(2) Remove punctuation
punctuation = dict.fromkeys(i for i in xrange(sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P'))
test3 = test2.map(lambda x: x.translate(punctuation))

#(3) Remove unicode text u'\u200b.  It's essentially a zero width blank character 
#    that is repeatedly found in the text.  It would change character counts, etc.
test4 = test3.map(lambda x: x.replace(u'\u200b',''))





