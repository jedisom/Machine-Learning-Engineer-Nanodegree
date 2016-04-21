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
X_raw = tidy_data.loc[:, :'text_length'].copy(deep = True)
#create_features creates features that can't be created once data has been split
# into training and test data.
X = create_features(X_raw.copy(deep = True))

#take log in order to deal with exponentially larger y data at beginning of dataset
y_raw = tidy_data.loc[:, 'secPc'].copy(deep = True)
y = tidy_data.loc[:, 'secPc'].apply(log)

#Create initial visualization of the data
plt.plot(X.loc[:, 'cum_char'], y_raw, "o")
plt.ylabel('Seconds per Character')
plt.xlabel('Cumulative Characters Read')
plt.title('Exploratory Plot of Chinese Characters Reading Speed')
plt.close()

#Print out some summary statistics
n = X.shape[0]
print 'There are %d total records in the dataset' % n
print 'The dataset includes a total of %d minutes of study time' % X.loc[n-1,'cum_time']
print 'There are %d total characters read in the dataset' % X.loc[n-1,'cum_char']
mean_speed = round((X.loc[n-1,'cum_time'] * 60.0) / X.loc[n-1,'cum_char'], 3)
print 'The mean reading speed over the entire dataset is %r seconds per character' % mean_speed

#Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#from feature_creation import find_topics
#X_train = find_topics(X_train, 10)

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
X_train_baseline = X_train.loc[:, 'ln_cum_char'].reshape((n_train,1))
clf = linear_model.LinearRegression(copy_X=True, fit_intercept=True)
random.seed = 1
baseline_scores = cross_validation.cross_val_score(clf, X_train_baseline, y_train, 
                                                  scoring='mean_squared_error', cv=10)  
#https://github.com/scikit-learn/scikit-learn/issues/2439
base_mean = baseline_scores.mean()*(-1) #output is negative and needs to be reversed
print("Baseline MSE Cross-Validation Mean: %0.4f" % base_mean)
clf.fit (X_train_baseline, y_train)

#show linear fit on cumulative time scatter plot
fit_line_X = [min(X_train_baseline), max(X_train_baseline)]
fit_line_y = clf.predict(fit_line_X)

plt.plot(fit_line_X, fit_line_y, "k", X_train.loc[:, 'ln_cum_char'], y_train, "o")
plt.ylabel('ln(Seconds per Character)')
plt.xlabel('ln(Cumulative Characters Read)')
plt.title('Baseline Fit to Chinese Characters Reading Speed Experience Curve')
m = str(round(clf.coef_[0],9))
b = str(round(clf.intercept_,4))
plt.text(3.0, 2.2, (r'y = ' + m + r' * x + ' + b))
plt.text(3.5, 2.0, (r'MSE = %0.4f' % base_mean))
plt.show()  

###Improve on baseliine by using features created from the text

##This section uses simple linear regression as sandbox to find features that
##will likely give a better fit
from feature_creation import find_topics
X_train = find_topics(X_train, 3)
                         
LinReg = linear_model.LinearRegression(copy_X=True, fit_intercept=True)
random.seed = 1
model = LinReg
feature_list = ('ln_cum_char', 'percent_seen', 'mean_days_since', 
                'mean_term_freq', 'norm_t1', 'norm_t2', 'norm_t3')  
X_train_sub = X_train.loc[:, feature_list]
char_count_scores = cross_validation.cross_val_score(model, X_train_sub, 
                                                     y_train, cv=10, 
                                                     scoring='mean_squared_error')                      
score_mean = char_count_scores.mean() * (-1)
print("New Model MSE Cross-Validation Mean: %0.4f" % score_mean)

#Add char count linear regression fit to the scatter plot for reference
LinReg.fit (X_train_sub, y_train)
fit_line_y = LinReg.predict(X_train_sub)
plt.plot(X_train.loc[:, 'cum_time'], fit_line_y, "x")

##This section explores different models with the feature set found to work 
## well in linear regression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

feature_list = ('ln_cum_char', 'percent_seen', 'mean_days_since', 
                'mean_term_freq', 'norm_t1', 'norm_t2', 'norm_t3')
X_train_sub = X_train.loc[:, feature_list]

RF_params = {'n_estimators': [100, 120, 150], 'max_features': ['auto', 'log2'],
             'max_depth': [9, 12, 15]}
BR_params = {'alpha_1': [3e-07, 1e-06, 3e-06], 'alpha_2': [3e-07, 1e-06, 3e-06],
             'lambda_1': [3e-07, 1e-06, 3e-06], 'lambda_2': [3e-07, 1e-06, 3e-06]}
Ridge_params = {'alpha': [0.1, 0.3, 1.0, 3.0, 10.0]}
DT_params = {'splitter': ['random', 'best'], 'max_features': ['auto', 'log2', None],
             'max_depth': [2,3,4]}
SV_params = {'kernel':('linear', 'rbf'), 'C':[0.3, 1.0, 3.0, 10.0]}
param_list = [RF_params, BR_params, Ridge_params, DT_params, SV_params]

RF_model = RandomForestRegressor(random_state = 1)
BR_model = BayesianRidge(copy_X = True)
Ridge_model = Ridge()
DT_model = DecisionTreeRegressor()
SV_model = SVR()
model_list = [RF_model, BR_model, Ridge_model, DT_model, SV_model]

MSE = make_scorer(score_func = mean_squared_error, greater_is_better = False) 
n = len(param_list)
for i in range(0, n):
    params = param_list[i]
    model = model_list[i]
    clf = GridSearchCV(estimator = model, param_grid = params, 
                       scoring = MSE, cv = 10)
    clf.fit(X_train_sub, y_train)
    print (clf.best_estimator_)
    print ("The best score for the model above is %0.4f" % (clf.best_score_ *(-1)))