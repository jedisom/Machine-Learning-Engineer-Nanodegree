# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 06:56:12 2016
@author: jed.isom

This script was created to take the raw copy/paste data set and turn it into
a tidy dataset.  Mostly, it takes the punctuation and extra spacing out of the 
text and then calculates the seconds per character metric that will be used
as the Y, output, variable for supervised learning.
"""
def tidy_up_data(filename):
    
    #import libraries needed
    import win32com.client
    import os
    import pandas as pd
    import datetime as dt
    import string
    import sys
    import unicodedata
    
    #I got a HUGE amount of help importing Chinese characters from Excel to python 
    #from these StackOverlow posts.  Other helpful sources are likewise referenced below.
    #http://stackoverflow.com/questions/29767898/how-to-read-excel-unicode-characters-using-python
    #http://stackoverflow.com/questions/5879754/run-excel-file-from-python-error
    
    #Setup port to Excel file where data is stored
    xl = win32com.client.gencache.EnsureDispatch('Excel.Application')
    xl.Workbooks.Open(os.path.abspath(filename))
    
    #Initialize dataframe where raw data will be written
    columns = ["date", "time_spent", "text_read"]
    tidy_data = pd.DataFrame(columns = columns)
    
    #Cycle through all rows with data and input data into tidy_data dataframe
    print ('Extracting raw data from Excel file...')    
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
    xl.Workbooks.Close()
    #print tidy_data.head()  #for debugging
    
    ###
    #Clean up text_read column
    ###
    
    #(1) Remove numbers and whitespace
    #http://stackoverflow.com/questions/12851791/removing-numbers-from-string
    print ('Removing numbers and whitespace...')    
    exclude = unicode(string.digits) + '\n' + ' '
    tidy_data.loc[:,'text_read'] = tidy_data.loc[:,'text_read'].map(lambda x: x.translate({ord(n): None for n in exclude}))
    
    #(2) Remove punctuation
    #http://stackoverflow.com/questions/11066400/remove-punctuation-from-unicode-formatted-strings
    print ('Removing punctuation...')    
    punctuation = dict.fromkeys(i for i in xrange(sys.maxunicode)
                          if unicodedata.category(unichr(i)).startswith('P'))
    tidy_data.loc[:,'text_read'] = tidy_data.loc[:,'text_read'].map(lambda x: x.translate(punctuation))
    
    #(3) Remove unicode text u'\u200b.  It's essentially a zero width blank character 
    #    that is repeatedly found in the text.  It would change character counts, etc.
    print ("Removing u'\u200b', a zero width blank character...")    
    tidy_data.loc[:,'text_read'] = tidy_data.loc[:,'text_read'].map(lambda x: x.replace(u'\u200b',''))
    
    #Add columns to dataframe for "text_read" length & seconds per character
    tidy_data.loc[:,'text_length'] = tidy_data.loc[:,'text_read'].map(lambda x: int(len(x)))
    tidy_data.loc[:,'secPc'] = tidy_data.loc[:, 'time_spent'] * 60.0 / tidy_data.loc[:, 'text_length']
    
    print ('Data is tidy and ready for analysis')
    return tidy_data
    #export to to .csv check against hand written character counts
    #tidy_data.loc[:,('date', 'text_length')].to_csv('tidy.csv', header = True)


