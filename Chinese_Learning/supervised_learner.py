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
base_estimator = linear_model.LinearRegression(copy_X=True, fit_intercept=True)
random.seed = 1
baseline_scores = cross_validation.cross_val_score(base_estimator, X_train_baseline, y_train, 
                                                  scoring='mean_squared_error', cv=10)  
#https://github.com/scikit-learn/scikit-learn/issues/2439
base_mean = baseline_scores.mean()*(-1) #output is negative and needs to be reversed
print("Baseline MSE Cross-Validation Mean: %0.4f" % base_mean)
base_estimator.fit (X_train_baseline, y_train)

#show linear fit on cumulative time scatter plot
fit_line_X = [min(X_train_baseline), max(X_train_baseline)]
fit_line_y = base_estimator.predict(fit_line_X)

plt.plot(fit_line_X, fit_line_y, "k", X_train.loc[:, 'ln_cum_char'], y_train, "o")
plt.ylabel('ln(Seconds per Character)')
plt.xlabel('ln(Cumulative Characters Read)')
plt.title('Baseline Fit to Chinese Characters Reading Speed Experience Curve')
m = str(round(base_estimator.coef_[0],9))
b = str(round(base_estimator.intercept_,4))
plt.text(3.0, 2.2, (r'y = ' + m + r' * x + ' + b))
plt.text(3.5, 2.0, (r'MSE = %0.4f' % base_mean))
plt.show()  

###Improve on baseline by using features created from the text

##This section uses simple linear regression as sandbox to find features that
##will likely give a better fit
from feature_creation import find_topics
X_train = find_topics(X_train, X_test, 3)
                         
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
plt.plot(X_train.loc[:, 'ln_cum_time'], fit_line_y, "x")
plt.close()

##This section explores different models with the feature set found to work 
## well in linear regression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

feature_list = ('ln_cum_char', 'percent_seen', 'mean_days_since', 
                'mean_term_freq', 'norm_t1', 'norm_t2', 'norm_t3') #
X_train_sub = X_train.loc[:, feature_list]

RF_params = {'n_estimators': [3, 5, 10, 20], 'max_features': ['auto', 'log2', None],
             'max_depth': [2,3,4]}

#RF_params = {'n_estimators': [100, 120, 150], 'max_features': ['auto', 'log2'],
#             'max_depth': [9, 12, 15]}
BR_params = {'alpha_1': [3e-07, 1e-06, 3e-06], 'alpha_2': [3e-07, 1e-06, 3e-06],
             'lambda_1': [3e-07, 1e-06, 3e-06], 'lambda_2': [3e-07, 1e-06, 3e-06]}
Ridge_params = {'alpha': [0.1, 0.3, 1.0, 3.0, 10.0]}
DT_params = {'splitter': ['random', 'best'], 'max_features': ['auto', 'log2', None],
             'max_depth': [2,3,4]}
n = len(feature_list)
SV_params = {'kernel':('linear', 'rbf'), 'C':[0.3, 1.0, 3.0, 10.0], 
             'gamma': [0.3/n, 1.0/n, 3.0/n]}
param_list = [RF_params, BR_params, Ridge_params, DT_params, SV_params]

RF_model = RandomForestRegressor(random_state = 1)
BR_model = BayesianRidge(copy_X = True)
Ridge_model = Ridge()
DT_model = DecisionTreeRegressor()
SV_model = SVR()
model_list = [RF_model, BR_model, Ridge_model, DT_model, SV_model]

MSE = make_scorer(score_func = mean_squared_error, greater_is_better = False) 
n = len(param_list)
best_MSE = 1000.0    
best_model = None
best_estimator = None
for i in range(0, n):
    params = param_list[i]
    model = model_list[i]
    
    clf = GridSearchCV(estimator = model, param_grid = params, 
                       scoring = MSE, cv = 10)
    clf.fit(X_train_sub, y_train)
    print (clf.best_estimator_)
    print ("The best score for the model above is %0.4f" % (clf.best_score_ *(-1)))
    if (clf.best_score_ *(-1)) < best_MSE:
        best_MSE = (clf.best_score_ *(-1))
        best_estimator = clf.best_estimator_

#This code is here to fine tune the random forest model
feature_list = ('ln_cum_char', 'percent_seen', 'mean_days_since', 
                'mean_term_freq', 'norm_t1', 'norm_t2', 'norm_t3') #
X_train_sub = X_train.loc[:, feature_list]
params = {'n_estimators': [100, 120, 150], 'max_features': ['auto', 'log2'],
             'max_depth': [9, 12, 15]}       
model = RF_model
clf = GridSearchCV(estimator = model, param_grid = params, 
                       scoring = MSE, cv = 10)
clf.fit(X_train_sub, y_train)
best_estimator = clf.best_estimator_
print (clf.best_estimator_)
print ("The best score for the model above is %0.4f" % (clf.best_score_ *(-1)))
    
###Compare the baseline model to the best predictor on the test set
from scipy.stats.mstats import normaltest
from scipy.stats import ttest_ind

n_test = X_test.shape[0]
X_test_baseline = X_test.loc[:, 'ln_cum_char'].reshape((n_test,1))
base_estimator = linear_model.LinearRegression(copy_X=True, fit_intercept=True)
random.seed = 1
base_estimator.fit (X_train_baseline, y_train) #Train model with original training set
baseline_test_scores = cross_validation.cross_val_score(base_estimator, X_test_baseline, y_test, 
                                                  scoring='mean_squared_error', cv=10)
p_base_normality = normaltest(baseline_test_scores)[1]                                                  
                                                  
feature_list = ('ln_cum_char', 'percent_seen', 'mean_days_since', 
                'mean_term_freq', 'norm_t1', 'norm_t2', 'norm_t3') #
X_test_sub = X_test.loc[:, feature_list]
best_test_scores = cross_validation.cross_val_score(best_estimator, X_test_sub, y_test, 
                                                  scoring='mean_squared_error', cv=10)
p_best_normality = normaltest(best_test_scores)[1]    
t_P_value = ttest_ind(baseline_test_scores, best_test_scores)[1]                                             
print "Normality test for baseline CV MSE gives a p-value of %0.4f" % p_base_normality
print "Normality test for best model's CV MSE gives a p-value of %0.4f" % p_best_normality
print "t-test for independece between baseline and best model gives a p-value of %0.4f" % t_P_value

#https://github.com/scikit-learn/scikit-learn/issues/2439
base_test_CVmean = baseline_test_scores.mean()*(-1) #output is negative and needs to be reversed    
base_test_CVstdev = baseline_test_scores.std()      


y_test_base = base_estimator.predict(X_test_baseline) #Estimate y with model created from training set
MSE_base = mean_squared_error(y_test, y_test_base) #MSE on test for model based on training set

#Best MSE on test set
y_test_best = best_estimator.predict(X_test.loc[:, feature_list])
MSE_best = mean_squared_error(y_test, y_test_best)

#Create plot (ln/ln) of Actual and Predicted Data
Raw = plt.plot(X_test.loc[:, 'ln_cum_char'], y_test, "o", label = 'Raw Data')
Base = plt.plot(X_test.loc[:, 'ln_cum_char'], y_test_base, "x", label = 'Baseline')
Best = plt.plot(X_test.loc[:, 'ln_cum_char'], y_test_best, "x", label = 'Best Model')
#plt.plot(X_test.loc[:, 'ln_cum_char'], y_test, "o", 
#         X_test.loc[:, 'ln_cum_char'], y_test_best, "x")
plt.ylabel('ln(Seconds per Character)')
plt.xlabel('ln(Cumulative Characters Read)')
plt.title('Baseline/Best Fit to Test Set Chinese Characters Reading Speed')
plt.legend()


ax = plt.subplot(111, xlabel='x', ylabel='y', title='title')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(20)
# Correlation plots in input features
#http://matplotlib.org/examples/pylab_examples/subplots_demo.html
n = len(feature_list)
f, axarr = plt.subplots(n, n)
for i in range(0, n):
    for j in range(0, n):
        axarr[i, j].scatter(X_train.loc[:,feature_list[i]],
                            X_train.loc[:,feature_list[j]], s = 20)#"o", 
        axarr[i, j].set_title("X: " + feature_list[i] + "; Y: " + feature_list[j])
        axarr[i, j].title.set_fontsize(8)

for i in range (0, n):
    plt.setp([a.get_xticklabels() for a in axarr[i, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, i]], visible=False)
#axarr[0, 0].plot(x, y)
#axarr[0, 0].set_title('Axis [0,0]')
#axarr[0, 1].scatter(x, y)
#axarr[0, 1].set_title('Axis [0,1]')
#axarr[1, 0].plot(x, y ** 2)
#axarr[1, 0].set_title('Axis [1,0]')
#axarr[1, 1].scatter(x, y ** 2)
#axarr[1, 1].set_title('Axis [1,1]')
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

#Create non-ln plot of Actual and Predicted Data
#from math import exp
#n_test = y_test.shape[0]
#y_test_exp = y_test.apply(exp)
#y_test_base_exp = y_test_base.apply(exp)

#Raw = plt.plot(X_test.loc[:, 'cum_char'], y_test_exp, "o", label = 'Raw Data')
#Base = plt.plot(X_test.loc[:, 'cum_char'], y_test_base.apply(exp), "x",
#                label = 'Baseline')
#Best = plt.plot(X_test.loc[:, 'cum_char'], y_test_best.apply(exp), "x", 
#                label = 'Best Model')
#plt.plot(X_test.loc[:, 'ln_cum_char'], y_test, "o", 
#         X_test.loc[:, 'ln_cum_char'], y_test_best, "x")
#plt.ylabel('Seconds per Character')
#plt.xlabel('Cumulative Characters Read')
#plt.title('Baseline & Best Fit to Chinese Characters Reading Speed')
#plt.legend()
