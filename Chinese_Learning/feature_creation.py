# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:57:32 2016
@author: jed.isom

This file takes the tidy X features (date, time spent, text, and character count)
and then creates derivative features that might allow the supervised learning
algorithm to more accurately predict the seconds per character metric.
"""

def Add_cumsums(df):
    print 'Calculating cumulative sums...'    
    from math import log
    
    #Add cumulative time read so far    
    df.loc[:,'cum_time'] = df.loc[:,'time_spent'].cumsum(axis = 0)
    df.loc[:,'ln_cum_time'] = df.loc[:,'cum_time'].apply(log)
    
    #Add cumulative chinese characters read so far
    df.loc[:,'cum_char'] = df.loc[:,'text_length'].cumsum(axis = 0)
    df.loc[:,'ln_cum_char'] = df.loc[:,'cum_char'].apply(log)
    
    return df

def normalize_features(df, feature_list):
    
    #normalize features from feature_list so they have a range from 0 to 1
    for feature in feature_list:
        n_feature = 'norm_' + feature        
        df.loc[:, n_feature] = ((df.loc[:, feature] - min(df.loc[:, feature])) / 
                             (max(df.loc[:, feature]) - min(df.loc[:, feature])))
                             
    return df 
    
def char_counts(df):
    #This function creates columns for each character and counts the times it
    #appears during each study session/day
    
    print 'Calculating time since character last read, etc...'    
    
    #Need to create corpus of characters found in all text_read    
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(decode_error = 'strict', analyzer = 'char')
    corpus = df.loc[:,'text_read']
    dtm = vectorizer.fit_transform(corpus)

    
    import numpy as np  
    from itertools import chain
    import datetime
    from scipy.sparse import csr_matrix
    
    n = df.shape[0]
    df.loc[:, 'percent_seen'] = 0.0
    df.loc[:, 'mean_days_since'] = 0.0
    df.loc[:, 'mean_term_freq'] = 0.0
    for i in range(1, n):   #cycle through all rows except first row
        ##Get percent of characters not seen in text so far        
        prior_non_zero = dtm[:i,:].nonzero()    #Find non-zero values in sparse matrix in (i-1) records
        before_chars = np.unique(prior_non_zero[1])  #Get list of all characters that have been seen so far
        current_chars = np.sort(dtm[i,:].nonzero()[1]) #Find non-zero characters in current record as column #'s    
        #http://stackoverflow.com/questions/28901311/numpy-find-index-of-elements-in-one-array-that-occur-in-another-array
        matching_current_index = np.where(np.in1d(current_chars, before_chars))[0]
        df.loc[i,'percent_seen'] = float(matching_current_index.shape[0])/float(current_chars.shape[0])
        
        ##Get mean days since characters last read (for those already seen in text)        
        #http://stackoverflow.com/questions/10252766/python-numpy-get-array-locations-of-a-list-of-values
        #http://stackoverflow.com/questions/11860476/how-to-unnest-a-nested-list         
        
        #gets list of tuple arrays (1 array per char in matching chars) where each array gives the indices of 
        #prior_non_zero where that character can be found        
        matching_chars = current_chars[matching_current_index]
        prior_array_indices = [np.where(prior_non_zero[1] == k) for k in list(matching_chars)]
        prior_array_indices = list(chain(*prior_array_indices))
        last_date_indices = map(lambda x: max(x), prior_array_indices)        
        last_date_rows = prior_non_zero[0][last_date_indices]
        current_date = df.loc[i,'date']
        days_since_seen = map(lambda x: current_date - x, df.loc[last_date_rows, 'date'])
        df.loc[i,'mean_days_since'] = (sum(days_since_seen, datetime.timedelta(0)).total_seconds()
                                            / 86400.0 / (len(days_since_seen)))
    
        ##Get mean frequency of document terms in the corpus so far
        #    NOT including the text read during the study session
        denominator = float(csr_matrix.sum(dtm[:i,:]))
        numerator = csr_matrix.sum(dtm[:i, matching_current_index])
        df.loc[i, 'mean_term_freq'] = numerator / denominator        
    
    #Normalize the current features
    norm_feat_list = ['cum_time', 'cum_char', 'mean_days_since']
    df = normalize_features(df, norm_feat_list)    
    
    #Create interaction terms with cumulative time and character count features
    df.loc[:,'timeXper_seen'] = df.loc[:, 'norm_cum_time'] *  df.loc[:,'percent_seen'] 
    df.loc[:,'timeXdays_since'] = df.loc[:, 'norm_cum_time'] *  df.loc[:,'norm_mean_days_since']
    df.loc[:,'timeXterm_freq'] = df.loc[:, 'norm_cum_time'] *  df.loc[:,'mean_term_freq']
    
    return df                                
    
def create_features(df):
    
    new_df = Add_cumsums(df)
    new_df = char_counts(new_df)
    return new_df
    
def find_topics(df_train, df_test, n_topics):
    
    #http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html    
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation    
    
    # Use tf (raw term count) features for LDA.
    print("Extracting character frequency features for topic modeling...")
    
    #Need to create a dtm with combined (train/test) vocabulary in columns
    n_train = df_train.shape[0]    
    df_combined = df_train.copy(deep = True).append(df_test.copy(deep = True))
    vectorizer = CountVectorizer(decode_error = 'strict', analyzer = 'char')
    corpus_combined = df_combined.loc[:,'text_read']
    dtm_combined = vectorizer.fit_transform(corpus_combined) 
    
    #split the train and test data again to ensure we only use test set for
    #supervised cross-validated learning
    dtm_train = dtm_combined[:n_train,:]
    dtm_test = dtm_combined[n_train:,:]
    
    print("Fitting LDA models with character frequency features...")
    #This requires sklearn.__version__ to be 0.17.X or greater    
    lda = LatentDirichletAllocation(n_topics=n_topics, learning_method='batch', 
                                    random_state=0)
    #fit to the training document term matrix
    lda.fit(dtm_train)
    
    #create topic 'names' and columns in dataframe    
    topic_names = []    
    for i in range(0, n_topics):
        name = 't' + str(i+1)        
        topic_names.append(name)
        df_train.loc[:, name] = 0.0
        df_test.loc[:, name] = 0.0
    
    df_train.loc[:, topic_names] = lda.transform(dtm_train)
    df_test.loc[:, topic_names] = lda.transform(dtm_test)
    
    #normalize these topic features
    df_train = normalize_features(df_train, topic_names)    
    df_test = normalize_features(df_test, topic_names)
    
    return df_train