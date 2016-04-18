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

#Import and clean up data
raw_filename = 'Raw_Chinese_Learning_Log.xlsx'
tidy_data = tidy_up_data(raw_filename)

#Create features to be used in supervised learning
X_raw = tidy_data.loc[:, :'text_length']
X = create_features(X_raw.copy(deep = True))

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
#plt.plot(X_train.loc[:, 'cum_char'], y_train, "o")
#plt.ylabel('ln(Seconds per Character)')
#plt.xlabel('Cumulative Characters Read')
#plt.title('Chinese Character Reading Speed Scatter Plot')
#plt.show()  

#plt.plot(X_train.loc[:, 'cum_time'], y_train, "o")
#plt.ylabel('ln(Seconds per Character)')
#plt.xlabel('Cumulative Time Spent Reading')
#plt.title('Chinese Character Reading Speed Scatter Plot')
#plt.show()  

###Create baseline model fit (linear regression of ln(y))
from sklearn import linear_model
from sklearn import cross_validation

#http://stackoverflow.com/questions/30813044/sklearn-found-arrays-with-inconsistent-numbers-of-samples-when-calling-linearre
n_train = X_train.shape[0]
X_train_baseline = X_train.loc[:, 'cum_time'].reshape((n_train,1))
clf = linear_model.LinearRegression(copy_X=True, fit_intercept=True)
random.seed = 1
baseline_scores = cross_validation.cross_val_score(clf, X_train_baseline, y_train, cv=10)  
print("Baseline R^2 Cross-Validation Mean: %0.4f" % baseline_scores.mean())
clf.fit (X_train_baseline, y_train)

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

###Improve on baseliine by using features created from the text
#from sklearn.grid_search import GridSearchCV
#sklearn.ensemble.RandomForestClassifier
#sklearn.neural_network.MLPRegressor
#sklearn.linear_model.BayesianRidge
#sklearn.tree.DecisionTreeRegressor
#sklearn.svm.SVR
#sklearn.linear_model.Ridge
#sklearn.linear_model.LinearRegression

#RSE = make_scorer(mean_squared_error, X = X_train_baseline, y = y_train)                          
LinReg = linear_model.LinearRegression(copy_X=True, fit_intercept=True)
random.seed = 1
X_train_counts = X_train.loc[:, ('cum_time', 'percent_seen', 'mean_days_since_seen')]
char_count_scores = cross_validation.cross_val_score(LinReg, X_train_counts, y_train, cv=10)  
print("Char Count R^2 Cross-Validation Mean: %0.4f" % char_count_scores.mean())

#Add char count linear regression fit to the scatter plot for reference
LinReg.fit (X_train_counts, y_train)
fit_line_y = LinReg.predict(X_train_counts)
plt.plot(X_train.loc[:, 'cum_time'], fit_line_y, "x")

#LinRegGrid = GridSearchCV(linear_model.LinearRegression)


