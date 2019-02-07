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


rnd_st = np.random.RandomState(31)


datapath = '~/gitRepo/Predicting_Employee_Attrition/data/'
outpath = '~/gitRepo/Predicting_Employee_Attrition/output/'

df = pd.read_csv(datapath+'processed_train.csv', delimiter=',', encoding='utf-8')

train = df.copy(deep=True)
train = train.drop(columns=['has_quit'])
target = df['has_quit']


# split data into train/test sets
# Import the train_test_split method


# Split data into train and test sets as well as for validation and testing
train, test, target_train, target_val = train_test_split(train, target, train_size= 0.70, random_state=rnd_st)
#train, test, target_train, target_val = StratifiedShuffleSplit(attrition_final, target, random_state=0)

log_reg = LogisticRegression()
log_reg.fit(train, target_train)
train_predict = log_reg.predict(train)
test_predict = log_reg.predict(test)
y_prob = log_reg.predict(train)
y_pred = np.where(y_prob > 0.5, 1, 0)
train_test_error(train_predict , test_predict)

oversampler=SMOTE(random_state=rnd_st)
smote_train, smote_target = oversampler.fit_sample(train,target)