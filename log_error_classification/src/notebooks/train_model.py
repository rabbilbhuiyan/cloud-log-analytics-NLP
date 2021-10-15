# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts/modeling//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# importing all the modules and libraries

from get_data import DataLoader
from parse_data import DataParser
from preprocess_data import DataPreprocess

# +
import pandas as pd
import numpy as np
from numpy import random
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn import metrics
from sklearn.metrics import precision_score


# -

class TrainModel():
    
    def __init__(self):
        self.data = None
        self.classifier = None
        self.vectorizer = None
        self.X_test = None
        self.y_test = None

    
    @staticmethod
    def load_data():
        
        data = DataPreprocess().create_label_encoder()
        X = data['processed_log']
        y = data['label_ID']
    
        return X, y
    
    # creating function for evaluation 

    @staticmethod # alternatively we can add 'self' as an argument for the object
    def evaluate_classifier(title, classifier, vectorizer, X_test, y_test):
        
        X_test_tfidf = vectorizer.transform(X_test)
        y_pred = classifier.predict(X_test_tfidf)

        precision = metrics.precision_score(y_test, y_pred, average= 'weighted')
        recall = metrics.recall_score(y_test, y_pred, average= 'weighted')
        f1 = metrics.f1_score(y_test, y_pred, average= 'weighted')

        print("%s\t%f\t%f\t%f\n" % (title, precision, recall, f1))

    
    # creating the training classifier

    def train_classifier(data):
        
        # load data
        X, y = TrainModel().load_data()
        
        # splitting the data into train test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

        # the object to turn data (text) into vectors
        vectorizer = TfidfVectorizer()

        # create doceument term matrix
        dtm = vectorizer.fit_transform(X_train)

        # train Naive Bayes classifier
        naive_bayes_classifier = MultinomialNB().fit(dtm, y_train)

        # evaluating the model accuracy  
        TrainModel().evaluate_classifier('Naive Bayes\tTRAIN\t', naive_bayes_classifier, vectorizer, X_train, y_train)

        TrainModel().evaluate_classifier('Naive Bayes\tTEST\t', naive_bayes_classifier, vectorizer, X_test, y_test)

        # store the classifier so we can call that
        clf_filename = 'naive_bayes_classifier.pkl'
        pickle.dump(naive_bayes_classifier, open(clf_filename, 'wb'))

        # store the vectorizer so we can transform to new data
        vec_filename = 'tfidf_vectorizer.pkl'
        pickle.dump(vectorizer, open(vec_filename, 'wb'))
        
        return naive_bayes_classifier

trainModelInstance = TrainModel()
result = trainModelInstance.train_classifier()
result


