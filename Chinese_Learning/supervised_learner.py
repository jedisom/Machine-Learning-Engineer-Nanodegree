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
from feature_creation import create_features, find_topics
import create_plots as plot    
from math import log  
#import numpy as np
import random
from sklearn.cross_validation import train_test_split
from sklearn import linear_model, cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from scipy.stats.mstats import normaltest
from scipy.stats import ttest_ind, pearsonr


def summary_stats (X):
    n = X.shape[0]
    print 'There are %d total records in the dataset' % n
    print 'The dataset includes a total of %d minutes of study time' % X.loc[n-1,'cum_time']
    print 'There are %d total characters read in the dataset' % X.loc[n-1,'cum_char']
    mean_speed = round((X.loc[n-1,'cum_time'] * 60.0) / X.loc[n-1,'cum_char'], 3)
    print 'The mean reading speed over the entire dataset is %r seconds per character' % mean_speed

def train_baseline(X, y):
    #http://stackoverflow.com/questions/30813044/sklearn-found-arrays-with-inconsistent-numbers-of-samples-when-calling-linearre
    
    n = X.shape[0]
    X_baseline = X.loc[:, 'ln_cum_char'].reshape((n,1))
    base_estimator = linear_model.LinearRegression(copy_X=True, fit_intercept=True)
    random.seed = 1
    baseline_scores = cross_validation.cross_val_score(base_estimator, X_baseline, 
                                             y, scoring='mean_squared_error', cv=10)  
    #https://github.com/scikit-learn/scikit-learn/issues/2439
    base_mean = baseline_scores.mean()*(-1) #output is negative and needs to be reversed
    print("Baseline MSE Cross-Validation Mean: %0.4f" % base_mean)
    base_estimator.fit (X_baseline, y)
    
    return (base_estimator, base_mean)


def try_linear_features(X, y):
    #This function is where I tried out all of the different features I created
    
    random.seed = 1
    LinReg = linear_model.LinearRegression(copy_X=True, fit_intercept=True)
    feature_list = ('ln_cum_char', 'percent_seen', 'mean_days_since', 
                    'mean_term_freq', 'norm_t1', 'norm_t2', 'norm_t3')
    X_sub = X.loc[:, feature_list]
    char_count_scores = cross_validation.cross_val_score(LinReg, X_sub, 
                              y, cv=10, scoring='mean_squared_error')                      
    score_mean = char_count_scores.mean() * (-1)
    print("New Model MSE Cross-Validation Mean: %0.4f" % score_mean)
    
    return (X_sub, LinReg)

def try_several_models(X, y):
    ##This section explores different models with the feature set found to work 
    ## well in linear regression
    feature_list = ('ln_cum_char', 'percent_seen', 'mean_days_since', 
                    'mean_term_freq', 'norm_t1', 'norm_t2', 'norm_t3') #
    X_sub = X.loc[:, feature_list]
    
    #Create list of models to explore
    RF_model = RandomForestRegressor(random_state = 1)
    BR_model = BayesianRidge(copy_X = True)
    Ridge_model = Ridge()
    DT_model = DecisionTreeRegressor()
    SV_model = SVR()
    model_list = [RF_model, BR_model, Ridge_model, DT_model, SV_model]
    
    #Define lists of parameters to explore in each model
    RF_params = {'n_estimators': [3, 5, 10, 20], 'max_features': ['auto', 'log2', None],
                 'max_depth': [2,3,4]}
    BR_params = {'alpha_1': [3e-07, 1e-06, 3e-06], 'alpha_2': [3e-07, 1e-06, 3e-06],
                 'lambda_1': [3e-07, 1e-06, 3e-06], 'lambda_2': [3e-07, 1e-06, 3e-06]}
    Ridge_params = {'alpha': [0.1, 0.3, 1.0, 3.0, 10.0]}
    DT_params = {'splitter': ['random', 'best'], 'max_features': ['auto', 'log2', None],
                 'max_depth': [2,3,4]}
    n = len(feature_list)
    SV_params = {'kernel':('linear', 'rbf'), 'C':[0.3, 1.0, 3.0, 10.0], 
                 'gamma': [0.3/n, 1.0/n, 3.0/n]}
    param_list = [RF_params, BR_params, Ridge_params, DT_params, SV_params]
    
    #Setup CV scoring system and initialize variables
    MSE = make_scorer(score_func = mean_squared_error, greater_is_better = False) 
    n = len(param_list)
    best_MSE = 1000.0    
    best_estimator = None
    
    #Try all of the models and their parameter values with GridSearchCV
    for i in range(0, n):
        params = param_list[i]
        model = model_list[i]
        
        clf = GridSearchCV(estimator = model, param_grid = params, 
                           scoring = MSE, cv = 10)
        clf.fit(X_sub, y)
        print (clf.best_estimator_)
        print ("The best score for the model above is %0.4f" % (clf.best_score_ *(-1)))
        if (clf.best_score_ *(-1)) < best_MSE:
            best_MSE = (clf.best_score_ *(-1))
            best_estimator = clf.best_estimator_
            
    return (best_estimator)

def fine_tune_random_forest(X, y):
    #This code is here to fine tune the random forest model
    feature_list = ('ln_cum_char', 'percent_seen', 'mean_days_since', 
                    'mean_term_freq', 'norm_t1', 'norm_t2', 'norm_t3') #
    X_sub = X.loc[:, feature_list]
    params = {'n_estimators': [100, 120, 150], 'max_features': ['auto', 'log2'],
                 'max_depth': [9, 12, 15]}       
    model = RandomForestRegressor(random_state = 1)
    MSE = make_scorer(score_func = mean_squared_error, greater_is_better = False) 
    clf = GridSearchCV(estimator = model, param_grid = params, 
                           scoring = MSE, cv = 10)
    clf.fit(X_sub, y)
    best_estimator = clf.best_estimator_
    print (clf.best_estimator_)
    print ("The best score for the model above is %0.4f" % (clf.best_score_ *(-1)))
    
    return best_estimator
    
def compare_best_to_baseline(X_train, y_train, X_test, y_test, base_estimator, 
                             best_estimator):
    ###Compare the baseline model to the best predictor on the test set
    #Use baseline model to get CV data on test set    
    random.seed = 1
    n_test = X_test.shape[0]
    X_test_base = X_test.loc[:, 'ln_cum_char'].reshape((n_test, 1))
    baseline_test_scores = cross_validation.cross_val_score(base_estimator, 
                    X_test_base, y_test, scoring='mean_squared_error', cv=10)
                                                             
    #Use best model to get CV data on test set
    feature_list = ('ln_cum_char', 'percent_seen', 'mean_days_since', 
                    'mean_term_freq', 'norm_t1', 'norm_t2', 'norm_t3') #
    X_test_sub = X_test.loc[:, feature_list]
    best_test_scores = cross_validation.cross_val_score(best_estimator, 
                    X_test_sub, y_test, scoring='mean_squared_error', cv=10)
    
    #Calculate statistics to compare samples from baseline and best model    
    p_base_normality = normaltest(baseline_test_scores)[1]    
    p_best_normality = normaltest(best_test_scores)[1]       
    corr_p_value = pearsonr(baseline_test_scores, best_test_scores) 
    t_P_value = ttest_ind(baseline_test_scores, best_test_scores)[1]                                             
    
    print "Normality test for baseline CV MSE gives a p-value of %0.4f" % p_base_normality
    print "Normality test for best model's CV MSE gives a p-value of %0.4f" % p_best_normality
    print '''The Pearson correlation coefficient between the baseline and best model
    scores is %0.4F, and the correlation p-value is %0.4F''' % (corr_p_value[0], corr_p_value[1])
    print "t-test for independece between baseline and best model gives a p-value of %0.4f" % t_P_value    

    y_test_base = base_estimator.predict(X_test_base) #Estimate y with model created from training set
    MSE_base = mean_squared_error(y_test, y_test_base) #MSE on test for model based on training set
    print "The non-CV MSE for the baseline is %0.4f" % MSE_base
    
    #Best MSE on test set
    y_test_best = best_estimator.predict(X_test_sub)
    MSE_best = mean_squared_error(y_test, y_test_best)
    print "The non-CV MSE for the best model is %0.4f" % MSE_best
    
    return (y_test_base, y_test_best)


def run():
        
    #Import and clean up data
    raw_filename = 'Raw_Chinese_Learning_Log.xlsx'
    tidy_data = tidy_up_data(raw_filename)
    
    #Create features to be used in supervised learning
    X_raw = tidy_data.loc[:, :'text_length'].copy(deep = True)
    X = create_features(X_raw.copy(deep = True))
    
    #take log in order to deal with exponentially larger y data at beginning of dataset
    y_raw = tidy_data.loc[:, 'secPc'].copy(deep = True)
    y = tidy_data.loc[:, 'secPc'].apply(log)

    #Create initial data visualization
    plot.create_initial_plot(X.loc[:, 'cum_char'], y_raw)
    
    #Print some summary statistics
    summary_stats (X)
    
    #Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    (base_estimator, base_mean) = train_baseline(X_train, y_train)    
    
    #Create baseline model plot
    base_plot = plot.create_baseline_plot(X_train.loc[:, 'ln_cum_char'], y_train, 
                                           base_estimator, base_mean)
    
    ###Improve on baseline by using features created from the text
    #Add topic modeling features to X 
    X_train = find_topics(X_train, X_test, 3)
    
    #try/train new features and plot them    
    fit_output = try_linear_features(X_train, y_train)
    plot.add_feature_fit_to_baseline(base_plot, fit_output[0], y_train, fit_output[1])
    
    feature_list = ('ln_cum_char', 'percent_seen', 'mean_days_since', 
                    'mean_term_freq', 'norm_t1', 'norm_t2', 'norm_t3')
    try_several_models(X, y, feature_list)  
    best_estimator = fine_tune_random_forest(X, y, feature_list)
    (y_base, y_best) = compare_best_to_baseline(X_train, y_train, X_test, y_test, 
                             base_estimator, best_estimator)
    
    plot.best_model(X_test, y_test, y_base, y_best)
    plot.feature_correlations(X, feature_list)

if __name__ == '__main__':
    run()