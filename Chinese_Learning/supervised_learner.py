# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:40:20 2016
@author: jed.isom

This script calls raw_to_tidy.py to clean up the raw data, then does feature 
creation, and finally does supervised machine learning to determine the best way
to predict how fast the human learner (the past version of myself in this case)
will read traditional Chinese characters measured in seconds per character.
"""

from raw_to_tidy import tidy_up_data
from feature_creation import create_features

#Import and clean up data
raw_filename = 'Raw_Chinese_Learning_Log.xlsx'
tidy_data = tidy_up_data(raw_filename)

#Create features to be used in supervised learning
#Can't use time spent or text_length as learning features because I had a habit
#of studying for 1/2 a day and the algorithm would 'learn' that quickly.  That's
#not what I want to 'learn'.  I want to know if my reading speed could have been
#predicted based on how far along the learning curve I am (days read, cumulative
#time spent, cumulative characters read, etc.) as well as other features in the
#text data like (average days since characters in the text were last read, etc.)
X_raw = tidy_data.loc[:, :'text_length']
X = create_features(X_raw)
Y = tidy_data.loc[:, 'secPc']

#Print out some summary statistics
print 'There are %d total records in the dataset' % X.shape[0]
print 'The dataset includes a total of %d minutes of study time' % X.iloc[973,4]
print 'There are %d total characters read in the dataset' % X.iloc[973,5]
mean_speed = round((X.iloc[973,4] * 60.0) / X.iloc[973,5], 3)
print 'The mean reading speed over the entire dataset is %r seconds per character' % mean_speed
#Split dataset into training and test sets

