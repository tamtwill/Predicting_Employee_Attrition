#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 11:18:12 2019

@author: Tamara Williams

*****  PLACEHOLDER
*****  once I've picked a candidate model, I'll flesh this out.
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



# test whether over sampling helps
# oversampler=SMOTE(random_state=rnd_st)
# smote_train, smote_target = oversampler.fit_sample(train,target)