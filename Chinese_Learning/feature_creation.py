# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:57:32 2016
@author: jed.isom

This file takes the tidy X features (date, time spent, text, and character count)
and then creates derivative features that might allow the supervised learning
algorithm to more accurately predict the seconds per character metric.
"""

def Add_cumsums(df):
    print 'Calculating cumulative sums...'    
    
    #Add cumulative time read so far    
    df.loc[:,'cum_time'] = df.loc[:,'time_spent'].cumsum(axis = 0)
    
    #Add cumulative chinese characters read so far
    df.loc[:,'cum_char'] = df.loc[:,'text_length'].cumsum(axis = 0)
    
    return df
    
def char_counts(df):
    #This function creates columns for each character and counts the times it
    #appears during each study session/day
    
    print 'Calculating time since character last read, etc...'    
    
    #Need to create corpus of characters found in all text_read    
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(decode_error = 'strict', analyzer = 'char')
    corpus = df.loc[:,'text_read']
    dtm = vectorizer.fit_transform(corpus)

    
    import numpy as np  
    from itertools import chain
    import datetime
    
    n = df.shape[0]
    df.loc[:,'percent_seen'] = 0.0
    df.loc[:,'mean_days_since_seen'] = 0.0
    for i in range(1, n):   #cycle through all rows except first row
        #Get percent of characters not seen in text so far        
        prior_non_zero = dtm[:i,:].nonzero()    #Find non-zero values in sparse matrix in (i-1) records
        before_chars = np.unique(prior_non_zero[1])  #Get list of all characters that have been seen so far
        current_chars = np.sort(dtm[i,:].nonzero()[1]) #Find non-zero characters in current record as column #'s    
        #http://stackoverflow.com/questions/28901311/numpy-find-index-of-elements-in-one-array-that-occur-in-another-array
        matching_current_index = np.where(np.in1d(current_chars, before_chars))[0]
        df.loc[i,'percent_seen'] = float(matching_current_index.shape[0])/float(current_chars.shape[0])
        
        #Get mean days since characters last read (for those already seen in text)        
        #http://stackoverflow.com/questions/10252766/python-numpy-get-array-locations-of-a-list-of-values
        #http://stackoverflow.com/questions/11860476/how-to-unnest-a-nested-list         
        #Gets list of tuple arrays (1 array per char in matching chars) where each array gives the indices of 
        #prior_non_zero where that character can be found        
        matching_chars = current_chars[matching_current_index]
        prior_array_indices = [np.where(prior_non_zero[1] == k) for k in list(matching_chars)]
        prior_array_indices = list(chain(*prior_array_indices))
        last_date_indices = map(lambda x: max(x), prior_array_indices)        
        last_date_rows = prior_non_zero[0][last_date_indices]
        current_date = df.loc[i,'date']
        days_since_seen = map(lambda x: current_date - x, df.loc[last_date_rows, 'date'])
        df.loc[i,'mean_days_since_seen'] = (sum(days_since_seen, datetime.timedelta(0)).total_seconds()
                                            / 86400.0 / (len(days_since_seen)))

    return df    
    
def create_features(df):
    
    new_df = Add_cumsums(df)
    new_df = char_counts(new_df)
    return new_df