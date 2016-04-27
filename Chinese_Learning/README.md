
# Learning from Chinese Learning Data

This repository holds a project that analyzes data from a book in Chinese that I read.  I couldn't really read Chinese before reading this book, so the data documents my Chinese reading learning process.  I digitized the dataset (it was originally a written log) and used supervised learning analysis to determine what features in the text I read could accurately predict my reading speed as I progressed along the Chinese learning experience curve.

##Creating the Dataset
I have 2 spiral bound notebooks that contain the notes I took while reading this Chinese book.  While studying, if I didn't know a Chinese character, I would write the pinyin, English and Chinese character down.  I did this because I thought the repitition would help me learn to recognize the characters, and help me learn to write Chinese characters.  I also started to keep track of the dates I read, the amount of time I spent reading during each study session, and where in the text I started and stopped each day.  This last bit of information was critical; this is how I transformed the hand written notes into a digital dataset.  The Chinese text I read can be found on this [website](https://www.lds.org/scriptures/bofm?lang=zho) (see any applicable copywrite information there).  

Each row of the dataset contains the following fields
- Date: This is the date of the study session
- Time_Spent: This is the time I spent studying on that day in minutes
- Text_Read: This is the text from the book I was able to read during the time I was studying.

##Project Report
`Capstone Project Report.docx` containts a full discussion of this project and can be found within this repository.  For a detailed explanation of the the data, analysis, results, and next steps please review this file

##Files Used and Required Packages
###Files
- `Chinese_Learning_Log.xlsx`: This is the raw data file
- `raw_to_tidy.py`: This python script includes functions that turn the raw data file into a tidy dataset for analysis
- `feature_creation.py`: This python script creates new features from the Chinese text to be used as inputs/predictors in the supvervised learning problem
- `supervised_learner.py`: This python script is the master script that calls functions from the other scripts above.  It is also where I implemented the supervised learning algorithms to fit a more accurate model to the dataset.

###Required Packages
The following python packages are used in this project.  Before attempting to run `supervised_learner.py` please make sure you have all of these packages installed, with the correct versions if specified.
- `win32com.client`
- `os`
- `numpy`
- `pandas`
- `string`
- `sys`
- `unicodedata`
- `itertools.chain`
- `datetime`
- `math.log`
- `matplotlib.pyplot`
- `random`
- `scipy.stats.mstats.normaltest`
- `scipy.stats.ttest_ind`
- `scipy.stats.pearsonr`
- `scipy.sparse.csr_matrix`

`sklearn.__version__` required to be 0.17.X or greater    
- `sklearn.cross_validation.train_test_split`
- `sklearn.linear_model`
- `sklearn.cross_validation`
- `sklearn.grid_search.GridSearchCV`
- `sklearn.metrics.mean_squared_error`
- `sklearn.metrics.make_scorer`
- `sklearn.ensemble.RandomForestRegressor`
- `sklearn.linear_model.BayesianRidge`
- `sklearn.linear_model.Ridge`
- `sklearn.tree.DecisionTreeRegressor`
- `sklearn.svm.SVR`
- `sklearn.feature_extraction.text.CountVectorizer`
- `sklearn.decomposition.LatentDirichletAllocation`

##Run

To run this project: 

1. fork/clone this repository to your local computer.  
2. Make sure you are in the top-level project directory Chinese_Learning/ (that contains this README). 
3. Run: `python supervised_learner.py` in the command line

##Udacity Assignment Guidelines
>###Capstone Project Guidelines
>Use the following questions in each section to keep a mental checklist of if you are meeting the standards and requirements for each rubric item your project is graded against. You can find these questions in their respective sections of the project report template [here](https://docs.google.com/document/d/1B-vEOscvfqctGEMHTFDS9Nw7aqcE2iuwPRfp0jK8nf4/pub?embedded=true).
>####Definition
>#####Project Overview
>- Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?
>- Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?

>#####Problem Statement
>- Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?
>- Have you thoroughly discussed how you will attempt to solve the problem?
>- Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?

>#####Metrics
>- Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?
>- Have you provided reasonable justification for the metrics chosen based on the problem and solution?

>####Results
>#####Model Evaluation and Validation
>- Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?
>- Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?
>- Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?
>- Can results found from the model be trusted?

>#####Justification
>- Are the final results found stronger than the benchmark result reported earlier?
>- Have you thoroughly analyzed and discussed the final solution?
>- Is the final solution significant enough to have solved the problem?

>####Conclusion
>#####Free-Form Visualization
>- Have you visualized a relevant or important quality about the problem, dataset, input data, or results?

##License

Software and Report Copyright(c) 2016 Jed Isom

For Rights and Use Information for the book/text used to create the 
`Raw_Chinese_Learning_Log.xlsx` for this project please see this [website](https://www.lds.org/legal/terms?lang=eng&_r=1).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software") excluding the 
book/text used to created `Raw_Chinese_Learning_Log.xlsx`, to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
