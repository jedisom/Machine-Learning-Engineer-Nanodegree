# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:57:32 2016
@author: jed.isom

This file takes the tidy X features (date, time spent, text, and character count)
and then creates derivative features that might allow the supervised learning
algorithm to more accurately predict the seconds per character metric.
"""

def Add_cumsums(df):
    #Add cumulative time read so far    
    df.loc[:,'cum_time'] = df.loc[:,'time_spent'].cumsum(axis = 0)
    
    #Add cumulative chinese characters read so far
    df.loc[:,'cum_char'] = df.loc[:,'text_length'].cumsum(axis = 0)
    
    return df
    
def char_counts(df):
    #This function creates columns for each character and counts the times it
    #appears during each study session/day
    
    #Need to create corpus of characters found in all text_read    
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(decode_error = 'strict', analyzer = 'char')
    corpus = df.loc[:,'text_read']
    dtm = vectorizer.fit_transform(corpus)

    #Get percent of characters not seen in text so far
    import numpy as np    
    #from scipy.sparse import csr_matrix
    
    n = df.shape[0]
    df.loc[:,'percent_seen'] = 0.0
    for i in range(1, n):   #cycle through all rows except first row
        prior_non_zero = dtm[:(i-1),:].nonzero()    #Find non-zero values in sparse matrix in (i-1) records
        before_chars = np.unique(prior_non_zero[1])  #Get list of all characters that have been seen so far
        
        #Find non-zero characters in current record as column #'s
        current_chars = np.sort(dtm[i,:].nonzero()[1])  
        
        #http://stackoverflow.com/questions/28901311/numpy-find-index-of-elements-in-one-array-that-occur-in-another-array
        matching_current_index = np.where(np.in1d(current_chars, before_chars))[0]
        df.loc[i,'percent_seen'] = float(matching_current_index.shape[0])/float(current_chars.shape[0])
        
    #Get mean days since characters last read (for those already seen in text)
        

    return df    
    
def create_features(df):
    
    new_df = Add_cumsums(df)
    #new_df = char_counts(new_df)
    return new_df