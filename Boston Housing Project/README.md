# Student Intervention Project

Initial Supervised Learning Project  - from the Udacity Machine Learning Engineer Nanodegree
This was the first project in the nanodegree program to get us started.  The first 2 sections of the README file below are directly from Udacity's project description

## Project Overview

>In this project, you will apply basic machine learning concepts on data collected for housing prices in the Boston, Massachusetts area to predict the selling price of a new home. You will first explore the data to obtain important features and descriptive statistics about the dataset. Next, you will properly split the data into testing and training subsets, and determine a suitable performance metric for this problem. You will then analyze performance graphs for a learning algorithm with varying parameters and training set sizes. This will enable you to pick the optimal model that best generalizes for unseen data. Finally, you will test this optimal model on a new sample and compare the predicted selling price to your statistics.

>The Boston housing market is highly competitive, and you want to be the best real estate agent in the area. To compete with your peers, you decide to leverage a few basic machine learning concepts to assist you and a client with finding the best selling price for their home. Luckily, you’ve come across the Boston Housing dataset which contains aggregated data on various features for houses in Greater Boston communities, including the median value of homes for each of those areas. Your task is to build an optimal model based on a statistical analysis with the tools available. This model will then be used to estimate the best selling price for your client’s home.

>For this assignment, you can find the boston_housing.ipynb file as a downloadable in the Resources section. You may also visit our Projects GitHub to have access to all of the projects available for this Nanodegree. While some code has already been implemented to get you started, you will need to implement additional functionality to successfully answer all of the questions included in the notebook. You can find the included questions for reference on the following slide. Unless requested, do not modify code that has already been included.

## Project Highlights

>This project is designed to get you acquainted to working with datasets in Python and applying basic machine learning techniques using NumPy and Scikit-Learn. Before being expected to use many of the available algorithms in the sklearn library, it will be helpful to first practice analyzing and interpreting the performance of your model.

>Things you will learn by completing this project:

> - How to use NumPy to investigate the latent features of a dataset.
> - How to analyze various learning performance plots for variance and bias.
> - How to determine the best-guess model for predictions from unseen data.
> - How to evaluate a model’s performance on unseen data using previous data.
 
## Language and libraries required

Python 2.7
Numpy 1.10
Sklearn 0.17
iPython Notebook (with iPython 4.0)

## Reviewing the completed project
The ipython notebook `boston_housing.ipynb` contains the completed project file with working code.  To view my version of the completed the ipython notebook, open `boston_housing.ipynb`.  `Boston_Housing Answers.pdf` contains the same answers to the questions given in .ipynb file, but in a PDF format.

## Dataset

The following description of the dataset was taken directly from Udacity's README file about this project: 
>The dataset used in this project is included with the scikit-learn library (sklearn.datasets.load_boston). You do not have to download it separately.

>It contains the following attributes for each housing area, including median value (which you will try to predict):

>* CRIM: per capita crime rate by town
>* ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
>* INDUS: proportion of non-retail business acres per town
>* CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
>* NOX: nitric oxides concentration (parts per 10 million)
>* RM: average number of rooms per dwelling
>* AGE: proportion of owner-occupied units built prior to 1940
>* DIS: weighted distances to five Boston employment centres
>* RAD: index of accessibility to radial highways
>* TAX: full-value property-tax rate per $10,000
>* PTRATIO: pupil-teacher ratio by town
>* B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
>* LSTAT: % lower status of the population
>* MEDV: Median value of owner-occupied homes in $1000's
