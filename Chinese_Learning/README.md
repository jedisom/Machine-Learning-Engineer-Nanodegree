
# Learning from Chinese Learning Data

This repository is going to hold a project I'm working on that analyzes data from a book in Chinese that I read.  I couldn't really read Chinese before reading this book, so the data documents my Chinese reading learning process.
There is more to come as I digitize the dataset (it's currently a written log) and start a supervised learning analysis on the dataset.

##Creating the Dataset
I have 2 spiral bound notebooks that contain my notes I took while working to read this Chinese book.  During my study, if I didn't recognize a word/phrase, I would write the pinyin, English and Chinese character down.  I did this, because I thought that the repitition would help me learn how to recognize the characters, and help me to learn to write Chinese characters.  Along with this, I also started to keep track of the dates I read, the amount of time I spent reading during each study session, and where in the text I started and stopped each day.  This last bit of information is critical to this dataset.  This is how I am transforming my hand written notes into a digital dataset.  The text I read can be found on this [website](https://www.lds.org/scriptures/bofm?lang=zho) (see any applicable copywrite information there).  

Each row of the dataset contains the following fields
- Date: This is the date of the study session
- Time_Spent: This is the time I spent studying on that day in minutes
- Text_Read: This is the text from the book I was able to read during the time I was studying.

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
