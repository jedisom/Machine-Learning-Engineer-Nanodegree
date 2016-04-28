# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 20:14:25 2016
@author: jed.isom

This python script creates plots based on the Chinese Learning Log
supervised learning dataset
"""
import matplotlib.pyplot as plt
import numpy as np

def create_initial_plot(X, y):
    plt.figure(1)    
    plt.plot(X, y, "o")
    plt.ylabel('Seconds per Character')
    plt.xlabel('Cumulative Characters Read')
    plt.title('Exploratory Plot of Chinese Characters Reading Speed')
    
def create_baseline_plot1(X, y, model, MSE):
    ##Create initial test set visualization; baseline learning curve model
    
    #Cumulative Characters Plot
    plt.figure(2)   
    plt.plot(X, y, "bo", label = 'Raw Data')
    plt.ylabel('ln(Seconds per Character)')
    plt.xlabel('ln(Cumulative Characters Read)')
    plt.title('Baseline Fit to Chinese Characters Reading Speed Experience Curve')
      
    #Cumulative Time Plot
    #plt.figure(3)    
    #plt.plot(X, y, "bo", label = 'Raw Data')
    #plt.ylabel('ln(Seconds per Character)')
    #plt.xlabel('ln(Cumulative Time Spent Reading)')
    #plt.title('Baseline Fit to Chinese Characters Reading Speed Experience Curve')
    
    #show linear fit on cumulative char scatter plot
    fit_line_X = np.array([min(X), max(X)])
    fit_line_y = model.predict(fit_line_X.reshape(-1,1))      
    plt.plot(fit_line_X, fit_line_y, "k", label = 'Fit Line')
    m = str(round(model.coef_[0],9))
    b = str(round(model.intercept_,4))
    plt.text(3.0, 2.2, (r'y = ' + m + r' * x + ' + b))
    plt.text(3.5, 2.0, (r'MSE = %0.4f' % MSE))    

def create_baseline_plot2(X, y, model, MSE):
    
    #show linear fit on cumulative time scatter plot
    fit_line_X = np.array([min(X), max(X)])
    fit_line_y = model.predict(fit_line_X.reshape(-1,1))    
    
    plt.figure(4)     
    plt.plot(fit_line_X, fit_line_y, "k", label = 'Fit Line')
    plt.plot(X, y, "bo", label = 'Raw Data')
    plt.ylabel('ln(Seconds per Character)')
    plt.xlabel('ln(Cumulative Characters Read)')
    plt.title('Baseline Fit to Chinese Characters Reading Speed Experience Curve')
    m = str(round(model.coef_[0],9))
    b = str(round(model.intercept_,4))
    plt.text(3.0, 2.2, (r'y = ' + m + r' * x + ' + b))
    plt.text(3.5, 2.0, (r'MSE = %0.4f' % MSE)) 
    
def add_feature_fit_to_baseline(X, y, model):
    #Add char count linear regression fit to the scatter plot for reference
    model.fit (X, y)
    fit_line_y = model.predict(X)
    plt.figure(2)    
    plt.plot(X.loc[:, 'ln_cum_char'], fit_line_y, "gx", label = 'New Features')
    plt.legend()
    #plt.close()

def best_model(X, y, y_base, y_best):
    #Create plot (ln/ln) of Actual and Predicted Data       
    plt.figure(5)    
    plt.plot(X.loc[:, 'ln_cum_char'], y, "o", label = 'Raw Data') #Raw = 
    plt.plot(X.loc[:, 'ln_cum_char'], y_base, "x", label = 'Baseline') #Base = 
    plt.plot(X.loc[:, 'ln_cum_char'], y_best, "x", label = 'Best Model') #Best = 
    plt.ylabel('ln(Seconds per Character)')
    plt.xlabel('ln(Cumulative Characters Read)')
    plt.title('Baseline/Best Fit to Test Set Chinese Characters Reading Speed')
    plt.legend()
    
def feature_correlations(X, feature_list):
    # Correlation plots of input features for free-form visualization
    #http://matplotlib.org/examples/pylab_examples/subplots_demo.html    
    n = len(feature_list)
    f, axarr = plt.subplots(n, n)
    for i in range(0, n):
        for j in range(0, n):
            axarr[i, j].scatter(X.loc[:,feature_list[i]],
                                X.loc[:,feature_list[j]], s = 20)#"o", 
            axarr[i, j].set_title("X: " + feature_list[i] + "; Y: " + feature_list[j])
            axarr[i, j].title.set_fontsize(8)
    
    #This removes the X and Y axis tick labels to clean up the visualization
    for i in range (0, n):
        plt.setp([a.get_xticklabels() for a in axarr[i, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in axarr[:, i]], visible=False)    
    
    
