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


## Importation de la base de données

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


## Prétraitement des données (Data Pre-proceesing)

## Evaluation de l'importance des termes contenus dans le document textuel reviewText ou summary

## Calcul des fréquences (tf) des mots afin de diminuer l'impact des mots 
## qui apparaissent dans beaucoup de documents tels que les pronoms (tf-idf)
from sklearn.feature_extraction.text import TfidfVectorizer 
tfidf = TfidfVectorizer()
tfidf.fit(df['reviewText']) 

# Pour apliquer sur les données de summary, il suffiet de remplacer 'reviewText' par 'summary'


## Cas 1: la méthode de Classification binaire

## Division de la variable prédictive overall en deux classes 0 si nombre d'étoiles égal à 1 ou 2
## 1 si nombre d'étoiles égal à 3, 4 ou 5.
## Ainsi notre nouvelle variable prédictive sera binaire et on l'appelera Positivity

import numpy as np
df.dropna(inplace=True)
df[df['overall'] != 3]
df['Positivity'] = np.where(df['overall'] > 3, 1, 0)


## Maintenat qu'on a notre nouvelle variable prédictive Positivity, nous allons diviser notre base en deux:
## base d'apprentissage qu'on appelera X_train et base test qu'on appelera y_train
## Pour cela, nous avons assigné 80% des données à la base d'apprentissage et 20% à la base test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], df['Positivity'], test_size=0.2, random_state = 0)
print('X_train first entry: \n\n', X_train[0])

## Visualisation des données brutes

target_count = df.Positivity.value_counts()

print('overall 0:', target_count[0])
print('overall 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (Positivity)');


## On va maintenant appliquer nos classificateurs sur les données brute déséquilibrées

## Pour former notre classificateur, nous devons transformer nos titres de mots en nombres,
## car les algorithmes ne savent travailler qu'avec des nombres.

## Pour faire cette transformation, nous allons utiliser CountVectorizer de sklearn. 
## Il s'agit d'une classe très simple pour convertir des mots en caractéristiques.
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer().fit(X_train)
vect


X_train_vectorized = vect.transform(X_train)
X_train_vectorized



## Construction des modèles à partir de pipepline

## Création de la pipepline

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

## Pour les fois où le pipepline prenait assez de temps à fonctionner,
## le modèle à été construit à la main de la manière suivante:

## a) Construction du modèle Naive Bayes
from sklearn.naive_bayes import MultinomialNB 
nb = MultinomialNB() 
nb.fit(X_train_vectorized, y_train)

## Nouvelles prédiction du modèle sur la base test ou base de validation
y_pred = nb.predict(vect.transform(X_test))


## Affichage des valeurs estimés des métriques: accuracy, f-score, matrice de confusion, la précision
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score

## Martice de confusion
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
## Accuracy
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

## F-score
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100)) 

## Précision
print("\nF1 Score: {:.2f}".format(precision_score(y_test, y_pred) * 100))

## Affichage groupé des rapports des métriques par rapport de classification
from sklearn.metrics import classification_report 
print(classification_report(y_test, y_pred)) 




## b) Construction du modèle Random forest

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




## b) Construction du modèle kernel SVM

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












## Cas 2: la méthode de Classification multi-classe


## base d'apprentissage 80% des données
## base test 20% des données

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], df['overall'], test_size=0.2, random_state = 0)
print('X_train first entry: \n\n', X_train[0])

## Visualisation des scores brutes pour les 5 niveau d'étoiles

plt.figure(figsize=(8,5))
x=df.overall.value_counts()
ax = sns.barplot(x.index, x.values)
plt.title("overall")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of categories', fontsize=12)


## On va maintenant appliquer nos classificateurs sur les données brute déséquilibrées

## Pour former notre classificateur, nous devons transformer nos titres de mots en nombres,
## car les algorithmes ne savent travailler qu'avec des nombres.

## Pour faire cette transformation, nous allons utiliser CountVectorizer de sklearn. 
## Il s'agit d'une classe très simple pour convertir des mots en caractéristiques.

## a) Construction du modèle Naive Bayes

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







## b) Construction du modèle Random forest

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






## c) Construction du modèle kernel SVM

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





















########### Equilibrons les données par sous-echantillonage et sur-échantillonnage


## Phase de preprocessing

## Sous-échantilonnage: Undersampling pour méthode binaire
## Création de la variable prédite binaire: Positivity

import numpy as np
df.dropna(inplace=True)
df[df['overall'] != 3]
df['Positivity'] = np.where(df['overall'] > 3, 1, 0)

## Fonction permettant de nettoyer la base de données
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

## Nettoyage de la colonne contenue le test 'reviewText' ou 'summary'
df['reveiwText'] = df['reveiwText'].map(lambda com : clean_text(com))    

#Début de la procédure de sous-échantillonnage (Undersampling)

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


## Création des corpus d’apprentissage et de test avec respectivement 80% en 
## apprentissage et 20% en validation ou test en utilisant utilisons la procédure 
## train_test_split du module sklearn.model_selection

from sklearn.feature_extraction.text import CountVectorizer
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2) 


## a) Construction du modèle Naive Bayes

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





## b) Construction du modèle Random forest

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







## c) Construction du modèle kernel SVM

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









## Sur-échantilonnage: Oversampling pour méthode binaire (reviewText, summary)
## Création de la variable prédite binaire: Positivity

import numpy as np
df.dropna(inplace=True)
df[df['overall'] != 3]
df['Positivity'] = np.where(df['overall'] > 3, 1, 0)

## Fonction permettant de nettoyer la base de données
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

## Nettoyage de la colonne contenue le test 'reviewText' ou 'summary'
df['reveiwText'] = df['reveiwText'].map(lambda com : clean_text(com))    

#Début du sous-échantillonnage (Undersampling)
from sklearn.model_selection import train_test_split X = df.reviewText
y = df.Positivity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.feature_extraction.text import CountVectorizer 
vect = CountVectorizer(max_features=1000, binary=True) 
X_train_vect = vect.fit_transform(X_train)

##Suréchantillonnage (Oversampling) en utilisant la fonction SMOTE
from sklearn.feature_extraction.text import CountVectorizer 
from imblearn.over_sampling import SMOTE

sm = SMOTE()

X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)

## affichage des données équilibrées
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
 


## b) Construction du modèle Random forest

from sklearn.ensemble import RandomForestClassifier

rd=RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)
rd.fit(X_train_res, y_train_res)
y_pred = rd.predict(X_test_vect)
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100)) 
print("\nF1 Score: {:.2f}".format(precision_score(y_test, y_pred) * 100, average='micro'))
print(classification_report(y_test, y_pred)) 


## c) Construction du modèle kernel SVM

from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0)
svm.fit(X_train_res, y_train_res)
y_pred = svm.predict(X_test_vect)
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100)) 
print("\nF1 Score: {:.2f}".format(precision_score(y_test, y_pred) * 100, average='micro'))
print(classification_report(y_test, y_pred))













## Sur-échantilonnage: Oversampling pour méthode multiclasse (reviewText, summary)
## Création de la variable prédite binaire: Positivity


## Fonction permettant de nettoyer la base de données
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

## Nettoyage de la colonne contenue le test 'reviewText' ou 'summary'
df['reveiwText'] = df['reveiwText'].map(lambda com : clean_text(com))    

#Début du sous-échantillonnage (Undersampling)
from sklearn.model_selection import train_test_split X = df.reviewText
y = df.overall
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.feature_extraction.text import CountVectorizer 
vect = CountVectorizer(max_features=1000, binary=True) 
X_train_vect = vect.fit_transform(X_train)

##Suréchantillonnage (Oversampling) en utilisant la fonction SMOTE
from sklearn.feature_extraction.text import CountVectorizer 
from imblearn.over_sampling import SMOTE

sm = SMOTE()

X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)

## affichage des données équilibrées
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
 


## b) Construction du modèle Random forest

from sklearn.ensemble import RandomForestClassifier

rd=RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)
rd.fit(X_train_res, y_train_res)
y_pred = rd.predict(X_test_vect)
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100)) 
print("\nF1 Score: {:.2f}".format(precision_score(y_test, y_pred) * 100, average='micro'))
print(classification_report(y_test, y_pred)) 


## c) Construction du modèle kernel SVM

from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0)
svm.fit(X_train_res, y_train_res)
y_pred = svm.predict(X_test_vect)
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100)) 
print("\nF1 Score: {:.2f}".format(precision_score(y_test, y_pred) * 100, average='micro'))
print(classification_report(y_test, y_pred))








######### Validation croisée

from sklearn.model_selection import cross_val_score
scores = cross_val_score(nb,  # convertion des text dans le modèle
                         df.reviewText,  # données brutes
                         df.Positivity,  # données de la variable prédictive
                         cv=10,  # assignation aléatoire des données en 10 parties : 9 pour l'apprentissage, 1 pour la validation
                         scoring='accuracy',  # choix de la métrique
                         n_jobs=-1,  # -1 = utilisation de tous les noyaux = rapide
                         )
print(scores)

print('Total scores classified:', len(df))
print('Score:', sum(scores)/len(scores))
