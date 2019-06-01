#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:27:32 2019

@author: charlottebimou
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 20:18:26 2019

@author: charlottebimou
"""

import pandas as pd 
import json
import gzip
import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns


## Importing the database

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g: 
    yield eval(l)
    
def getDF(path): 
  i=0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('/Users/charlottebimou/Documents Local/Projet/reviews_CDs_and_Vinyl_5.json.gz')


## Data Pre-processing

## Evaluation of the importance of the terms contained in the textual document reviewText or summary

## Calculation of word frequencies (tf) to reduce the impact of words 
## that appear in many documents such as pronouns (tf-idf)

from sklearn.feature_extraction.text import TfidfVectorizer 
tfidf = TfidfVectorizer()
tfidf.fit(df['reviewText'])


# To apply to summary data, simply replace "reviewText" with "summary".


## Case 1: The Binary Classification Method

## Division of the predictive variable overall into two classes 0 if the number of stars is 1 or 2
## 1 if number of stars equal to 3, 4 or 5.
## So our new predictive variable will be binary and we will call it Positivity


import numpy as np
df.dropna(inplace=True)
df[df['overall'] != 3]
df['Positivity'] = np.where(df['overall'] > 3, 1, 0)


## Now that we have our new predictive variable Positivity, we will divide our database in two:
## learning base that we will call X_train and test base that we will call y_train
## To do this, we assigned 80% of the data to the learning database and 20% to the test database


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], df['Positivity'], test_size=0.2, random_state = 0)
print('X_train first entry: \n\n', X_train[0])

## Visualization of raw data

target_count = df.Positivity.value_counts()

print('overall 0:', target_count[0])
print('overall 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (Positivity)');


## We will now apply our classifiers to unbalanced raw data

## To form our classifier, we must transform our word titles into numbers,
## because algorithms can only work with numbers.

## To do this transformation, we will use CountVectorizer from sklearn. 
## This is a very simple class to convert words into characteristics.


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer().fit(X_train)
vect


X_train_vectorized = vect.transform(X_train)
X_train_vectorized



## Building models from pipepline

## Creation of the pipepline

import logging
import pandas as pd
import numpy as np
from numpy import random
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer



## #Naive Bayes
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))

## For the times when the pipepline took long enough to work,
## the model was built by hand as follows:

## a) Construction of the Naive Bayes model

from sklearn.naive_bayes import MultinomialNB 
nb = MultinomialNB() 
nb.fit(X_train_vectorized, y_train)

## New model prediction on the test or validation basis

y_pred = nb.predict(vect.transform(X_test))


## Display of the estimated values of the metrics: accuracy, f-score, confusion matrix, precision

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score

##  confusion matrix

print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
## Accuracy
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

## F-score
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100)) 

## Précision
print("\nF1 Score: {:.2f}".format(precision_score(y_test, y_pred) * 100))

## Grouped display of metric reports in relation to classification report

from sklearn.metrics import classification_report 
print(classification_report(y_test, y_pred)) 




## b) Random forest construction

#Random forest
from sklearn.ensemble import RandomForestClassifier

rd = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)),
              ])
rd.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = rd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))




## b) kernel SVM construction

#SVM
from sklearn.svm import SVC

svm= Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', SVC(kernel='rbf', random_state=0)),
              ])
svm.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = svm.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))












## Case 2: The Multi-class Classification method


## learning base 80% of data
## test database 20% of the data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], df['overall'], test_size=0.2, random_state = 0)
print('X_train first entry: \n\n', X_train[0])

## Display of raw scores for the 5 star levels


plt.figure(figsize=(8,5))
x=df.overall.value_counts()
ax = sns.barplot(x.index, x.values)
plt.title("overall")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of categories', fontsize=12)


## We will now apply our classifiers to unbalanced raw data

## To form our classifier, we must transform our word titles into numbers,
## because algorithms can only work with numbers.

## To do this transformation, we will use CountVectorizer from sklearn. 
## This is a very simple class to convert words into characteristics.

## a) Naive Bayes construction

#Naive Bayes
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))







## b) Random forest construction

#Random forest
from sklearn.ensemble import RandomForestClassifier

rd = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)),
              ])
rd.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = rd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))






## c) kernel SVM construction

#SVM
from sklearn.svm import SVC

svm= Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', SVC(kernel='rbf', random_state=0)),
              ])
svm.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = svm.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))





















## Let's balance the data by sub-sampling and oversampling


## Preprocessing phase

## Subsampling: Undersampling for binary method
## Creation of the binary predicted variable: Positivity

import numpy as np
df.dropna(inplace=True)
df[df['overall'] != 3]
df['Positivity'] = np.where(df['overall'] > 3, 1, 0)

## Function to clean the database
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

## Cleaning the column contained in the test 'reviewText' or 'summary'
df['reveiwText'] = df['reveiwText'].map(lambda com : clean_text(com))    

##Beginning of the sub-sampling procedure (Undersampling)

score_bas = len(df[df['Positivity'] == 0])
score_haut_indices = df[df.Positivity == 1].index
random_indices = np.random.choice(score_haut_indices,score_bas) 
score_haut_indices = df[df.Positivity == 0].index
under_sample_indices = np.concatenate([score_haut_indices,random_indices]) 
under_sample = df.loc[under_sample_indices]
print('Random under-sampling:')
print(under_sample.Positivity.value_counts())

from sklearn.model_selection import train_test_split 
X = under_sample['reviewText']
y = under_sample['Positivity']


## Creation of the learning and test corpuses with respectively 80% in 
## learning and 20% in validation or test using the procedure 
## train_test_split of the sklearn.model_selection module

from sklearn.feature_extraction.text import CountVectorizer
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2) 


## a) Naive Bayes construction

#Naive Bayes
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))





## b) Random forest construction

#Random forest
from sklearn.ensemble import RandomForestClassifier

rd = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)),
              ])
rd.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = rd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))







## c) kernel SVM construction

#SVM
from sklearn.svm import SVC

svm= Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', SVC(kernel='rbf', random_state=0)),
              ])
svm.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = svm.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))









## Oversampling for binary model (reviewText, summary)
## Creating the binary predicted variable: Positivity

import numpy as np
df.dropna(inplace=True)
df[df['overall'] != 3]
df['Positivity'] = np.where(df['overall'] > 3, 1, 0)

## Function to clean the database
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

## Cleaning the column contained in the test 'reviewText' or 'summary'
df['reveiwText'] = df['reveiwText'].map(lambda com : clean_text(com))    

#Undersampling
from sklearn.model_selection import train_test_split X = df.reviewText
y = df.Positivity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.feature_extraction.text import CountVectorizer 
vect = CountVectorizer(max_features=1000, binary=True) 
X_train_vect = vect.fit_transform(X_train)

##Oversampling using SMOTE function
from sklearn.feature_extraction.text import CountVectorizer 
from imblearn.over_sampling import SMOTE

sm = SMOTE()

X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)

## display of balanced data
unique, counts = np.unique(y_train_res, return_counts=True) 
print(list(zip(unique, counts)))

# a) Classifier Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_res, y_train_res)
X_test_vect = vect.transform(X_test)
y_pred = nb.predict(X_test_vect)
from sklearn.metrics import accuracy_score, f1_score, confusion_mat rix, precision_score
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 1 00))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred, average= 'macro') * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred, average= 'micro') * 100))
print("\nF1 Score: {:.2f}".format(precision_score(y_test, y_pred, a verage='micro') * 100))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# To show main classification report
from sklearn.metrics import classification_report 
print(classification_report(y_test, y_pred))
 


## b) Random forest model construction

from sklearn.ensemble import RandomForestClassifier

rd=RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)
rd.fit(X_train_res, y_train_res)
y_pred = rd.predict(X_test_vect)
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100)) 
print("\nF1 Score: {:.2f}".format(precision_score(y_test, y_pred) * 100, average='micro'))
print(classification_report(y_test, y_pred)) 


## c) kernel SVM model construction

from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0)
svm.fit(X_train_res, y_train_res)
y_pred = svm.predict(X_test_vect)
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100)) 
print("\nF1 Score: {:.2f}".format(precision_score(y_test, y_pred) * 100, average='micro'))
print(classification_report(y_test, y_pred))













## Over-sampling: Oversampling for multiclass method (reviewText, summary)
## Creation of the binary predicted variable: Positivity


## Function to clean the database
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

## cleaning of the column contained in the test 'reviewText' ou 'summary'
df['reveiwText'] = df['reveiwText'].map(lambda com : clean_text(com))    

# Undersampling
from sklearn.model_selection import train_test_split X = df.reviewText
y = df.overall
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.feature_extraction.text import CountVectorizer 
vect = CountVectorizer(max_features=1000, binary=True) 
X_train_vect = vect.fit_transform(X_train)

## Oversampling using SMOTE function
from sklearn.feature_extraction.text import CountVectorizer 
from imblearn.over_sampling import SMOTE

sm = SMOTE()

X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)

## display of balanced data
unique, counts = np.unique(y_train_res, return_counts=True) 
print(list(zip(unique, counts)))

# a) Classifier Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_res, y_train_res)
X_test_vect = vect.transform(X_test)
y_pred = nb.predict(X_test_vect)
from sklearn.metrics import accuracy_score, f1_score, confusion_mat rix, precision_score
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 1 00))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred, average= 'macro') * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred, average= 'micro') * 100))
print("\nF1 Score: {:.2f}".format(precision_score(y_test, y_pred, a verage='micro') * 100))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# To show main classification report
from sklearn.metrics import classification_report 
print(classification_report(y_test, y_pred))
 


## b) Random forest construction

from sklearn.ensemble import RandomForestClassifier

rd=RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)
rd.fit(X_train_res, y_train_res)
y_pred = rd.predict(X_test_vect)
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100)) 
print("\nF1 Score: {:.2f}".format(precision_score(y_test, y_pred) * 100, average='micro'))
print(classification_report(y_test, y_pred)) 


## c) kernel SVM construction

from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0)
svm.fit(X_train_res, y_train_res)
y_pred = svm.predict(X_test_vect)
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100)) 
print("\nF1 Score: {:.2f}".format(precision_score(y_test, y_pred) * 100, average='micro'))
print(classification_report(y_test, y_pred))








######### Cross Validation 

from sklearn.model_selection import cross_val_score
scores = cross_val_score(nb,  # text conversion in the model
                         df.reviewText,  # raw data
                         df.Positivity,  # data of the predictive variable
                         cv=10,  # random assignment of data in 10 parts: 9 for learning, 1 for validation
                         scoring='accuracy',  # choice of metric
                         n_jobs=-1,  # -1 = use of all cores = fast
                         )
print(scores)

print('Total scores classified:', len(df))
print('Score:', sum(scores)/len(scores))
