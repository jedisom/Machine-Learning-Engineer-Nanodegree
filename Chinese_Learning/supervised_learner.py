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
from math import log
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random

#import plotly.plotly as py

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

#take log in order to deal with exponentially larger y data at beginning of dataset
y = tidy_data.loc[:, 'secPc'].apply(log)

#Print out some summary statistics
n = X.shape[0]
print 'There are %d total records in the dataset' % n
print 'The dataset includes a total of %d minutes of study time' % X.iloc[n-1,4]
print 'There are %d total characters read in the dataset' % X.iloc[n-1,5]
mean_speed = round((X.iloc[n-1,4] * 60.0) / X.iloc[n-1,5], 3)
print 'The mean reading speed over the entire dataset is %r seconds per character' % mean_speed

#Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Create initial test set visualization; baseline learning curve model
plt.plot(X_train.loc[:, 'cum_char'], y_train, "o")
plt.ylabel('ln(Seconds per Character)')
plt.xlabel('Cumulative Characters Read')
plt.title('Chinese Character Reading Speed Scatter Plot')
plt.show()  

plt.plot(X_train.loc[:, 'cum_time'], y_train, "o")
plt.ylabel('ln(Seconds per Character)')
plt.xlabel('Cumulative Time Spent Reading')
plt.title('Chinese Character Reading Speed Scatter Plot')
plt.show()  

#Create baseline model fit (linear regression of ln(y))
from sklearn import linear_model
#from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import cross_validation
#from sklearn.grid_search import GridSearchCV

#http://stackoverflow.com/questions/30813044/sklearn-found-arrays-with-inconsistent-numbers-of-samples-when-calling-linearre

n_train = X_train.shape[0]
X_train_baseline = X_train.loc[:, 'cum_time'].reshape((n_train,1))
clf = linear_model.LinearRegression(copy_X=True, fit_intercept=True)
random.seed = 1
baseline_scores = cross_validation.cross_val_score(clf, X_train_baseline, y_train, cv=10)  
print("Baseline R^2 Cross-Validation Mean: %0.4f" % baseline_scores.mean())
clf.fit (X_train_baseline, y_train)

#print clf.coef_
#print clf.intercept_

#show linear fit on cumulative time scatter plot
fit_line_X = [min(X_train_baseline), max(X_train_baseline)]
fit_line_y = clf.predict(fit_line_X)

plt.plot(fit_line_X, fit_line_y, "k", X_train.loc[:, 'cum_time'], y_train, "o")
plt.ylabel('ln(Seconds per Character)')
plt.xlabel('Cumulative Time Spent Reading')
plt.title('Baseline Fit to Chinese Characters Reading Speed Experienc Curve')
m = str(round(clf.coef_[0],9))
b = str(round(clf.intercept_,4))
plt.text(7500, 3.5, (r'y = ' + m + r' * x + ' + b))
plt.text(10000, 3.3, (r'R^2 = %0.4f' %baseline_scores.mean()))
plt.show()  

#RSE = make_scorer(mean_squared_error, X = X_train_baseline, y = y_train)                          
#grid = GridSearchCV(linear_model.LinearRegression, scoring = RSE)

# draw vertical line from (70,100) to (70, 250)
#plt.plot([70, 70], [100, 250], 'k-', lw=2)

# draw diagonal line from (70, 90) to (90, 200)
#plt.plot([70, 90], [90, 200], 'k-')

#plot_url = py.plot_mpl(line, filename='mpl-docs/add-line')
