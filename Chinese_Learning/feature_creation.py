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
    
#def char_counts(df):
    #This function creates columns for each character and counts the times it
    #appears during each study session/day
    
#    import nltk.tokenize
    #Need to create corpus of characters found in all text_read
    
def create_features(df):
    
    new_df = Add_cumsums(df)
    #new_df = char_counts(new_df)
    return new_df