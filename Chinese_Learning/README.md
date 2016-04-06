
# Learning from Chinese Learning Data

This repository is going to hold a project I'm working on that analyzes data from a book in Chinese that I read.  I couldn't really read Chinese before reading this book, so the data documents my Chinese reading learning process.
There is more to come as I digitize the dataset (it's currently a written log) and start a supervised learning analysis on the dataset.

##Creating the Dataset
I have 2 spiral bound notebooks that contain my notes I took while working to read this Chinese book.  During my study, if I didn't recognize a word/phrase, I would write the pinyin, English and Chinese character down.  I did this, because I thought that the repitition would help me learn how to recognize the characters, and help me to learn to write Chinese characters.  Along with this, I also started to keep track of the dates I read, the amount of time I spent reading during each study session, and where in the text I started and stopped each day.  This last bit of information is critical to this dataset.  This is how I am transforming my hand written notes into a digital dataset.  The text I read can be found on this (website)[https://www.lds.org/scriptures/bofm?lang=zho] (see any applicable copywrite information there).  

Each row of the dataset contains the following fields
- Date: This is the date of the study session
- Time_Spent: This is the time I spent studying on that day in minutes
- Text_Read: This is the text from the book I was able to read during the time I was studying.

##MIT License

Copyright (c) 2016 Jed Isom

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
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