# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 20:14:25 2016
@author: jed.isom

This python script creates plots based on the Chinese Learning Log
supervised learning dataset
"""
import matplotlib.pyplot as plt

def create_initial_plot(X, y):
    plt.plot(X, y, "o")
    plt.ylabel('Seconds per Character')
    plt.xlabel('Cumulative Characters Read')
    plt.title('Exploratory Plot of Chinese Characters Reading Speed')
    
def create_baseline_plot(X, y, model, MSE):
    #Create initial test set visualization; baseline learning curve model
    fig = plt.figure()    
    plt.plot(X, y, "o")
    plt.ylabel('ln(Seconds per Character)')
    plt.xlabel('Cumulative Characters Read')
    plt.title('Chinese Character Reading Speed Scatter Plot')
    #plt.close()  
    
    #plt.plot(X_train.loc[:, 'cum_time'], y_train, "o")
    #plt.ylabel('ln(Seconds per Character)')
    #plt.xlabel('Cumulative Time Spent Reading')
    #plt.title('Chinese Character Reading Speed Scatter Plot')
    #plt.close()  
    
    #show linear fit on cumulative time scatter plot
    fit_line_X = [min(X), max(X)]
    fit_line_y = model.predict(fit_line_X)    
    
    plt.plot(fit_line_X, fit_line_y, "k", label = 'Fit Line')
    plt.plot(X, y, "o", label = 'Raw Data')
    plt.ylabel('ln(Seconds per Character)')
    plt.xlabel('ln(Cumulative Characters Read)')
    plt.title('Baseline Fit to Chinese Characters Reading Speed Experience Curve')
    m = str(round(model.coef_[0],9))
    b = str(round(model.intercept_,4))
    plt.text(3.0, 2.2, (r'y = ' + m + r' * x + ' + b))
    plt.text(3.5, 2.0, (r'MSE = %0.4f' % MSE)) 
    
    return fig
    
def add_feature_fit_to_baseline(fig, X, y, model):
    #Add char count linear regression fit to the scatter plot for reference
    model.fit (X, y)
    fit_line_y = model.predict(X)
    fig.plot(X.loc[:, 'ln_cum_char'], fit_line_y, "x", label = 'New Features')
    fig.legend()
    #plt.close()

def best_model(X, y, y_base, y_best):
    #Create plot (ln/ln) of Actual and Predicted Data
    Raw = plt.plot(X.loc[:, 'ln_cum_char'], y, "o", label = 'Raw Data')
    Base = plt.plot(X.loc[:, 'ln_cum_char'], y_base, "x", label = 'Baseline')
    Best = plt.plot(X.loc[:, 'ln_cum_char'], y_best, "x", label = 'Best Model')
    plt.ylabel('ln(Seconds per Character)')
    plt.xlabel('ln(Cumulative Characters Read)')
    plt.title('Baseline/Best Fit to Test Set Chinese Characters Reading Speed')
    plt.legend()
    plt.close()

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
    
    
