#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 11:18:12 2019

@author: Tamara Williams
"""


import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation, metrics 
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, f1_score


RND_ST = np.random.RandomState(31)


datapath = '~/gitRepo/Predicting_Employee_Attrition/data/'
outpath = '~/gitRepo/Predicting_Employee_Attrition/output/'
N_FOLDS = 10
RND_ST = 31

df = pd.read_csv(datapath+'processed_train.csv', delimiter=',', encoding='utf-8')

train = df.copy(deep=True)
train = train.drop(columns=['has_quit'])
target = df['has_quit']

## from https://www.programcreek.com/python/example/91149/sklearn.model_selection.StratifiedShuffleSplit
## 
# def validate(data, labels):   
#     '''
#     Ten-fold cross-validation with stratified sampling.
#     '''
#     accuracy_scores = []
#     precision_scores = []
#     recall_scores = []
#     f1_scores = []

#     stsp = StratifiedShuffleSplit(n_splits=N_FOLDS)
#     for train_index, test_index in stsp.split(data, labels):
#         x_train, x_test = data[train_index], data[test_index]
#         y_train, y_test = labels[train_index], labels[test_index]
#         clf.fit(x_train, y_train)
#         y_pred = clf.predict(x_test)
#         accuracy_scores.append(accuracy_score(y_test, y_pred))
#         precision_scores.append(precision_score(y_test, y_pred))
#         recall_scores.append(recall_score(y_test, y_pred))
#         f1_scores.append(f1_score(y_test, y_pred))

#     print('Accuracy', np.mean(accuracy_scores))
#     print('Precision', np.mean(precision_scores))
#     print('Recall', np.mean(recall_scores))
#     print('F1-measure', np.mean(f1_scores)) 





# test whether over sampling helps
# oversampler=SMOTE(random_state=rnd_st)
# smote_train, smote_target = oversampler.fit_sample(train,target)