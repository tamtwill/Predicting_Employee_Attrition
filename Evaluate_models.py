#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 11:18:12 2019

@author: Tamara Williams

For something like attrition, do you want a classifier to tell you yes/no, or are you looking for probability of leaving? 
Which is more useful to a business, or do you want to look at both?

"""


import pandas as pd
import numpy as np



from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor,\
 AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit


from matplotlib import pyplot as plt


RND_ST = np.random.RandomState(31)

# initialize various system variables
RANDOM_SEED = 13
SET_FIT_INTERCEPT=True

# set the number of folds for cross-validation
N_FOLDS = 10
#N_FOLDS = 1

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 

pd.set_option('display.max_columns', 10)

datapath = '~/gitRepo/Predicting_Employee_Attrition/data/'
outpath = '~/gitRepo/Predicting_Employee_Attrition/output/'


train = pd.read_csv(datapath+'processed_train.csv', delimiter=',', encoding='utf-8')
cols = train.columns.tolist()
cols.insert(0, cols.pop(cols.index('has_quit')))
train = train.loc[:, cols]

train_data = train.copy(deep=True)
train_data = train_data.drop(columns=['has_quit'])
target = train['has_quit']


# Build and cross-validate regression models
#--------------------------------------------------
# Setup the list of models to look at, let's try a range

reg_methods = ['LogisiticRegression', 'Ridge', 'Lasso', 
          'ElasticNet', 'BaggingRegressor', 
          'RandomForest', 'AdaBoost','GradientBoosting 1.0','GradientBoosting .1', 
          'Extra Trees', 'BernoulliNB']

regress_list = [LogisticRegression(fit_intercept = SET_FIT_INTERCEPT), 
               Ridge(alpha = 1, solver = 'cholesky', 
                     fit_intercept = SET_FIT_INTERCEPT, 
                     normalize = False, 
                     random_state = RANDOM_SEED),
               Lasso(alpha = 0.1, max_iter=10000, tol=0.01, 
                     fit_intercept = SET_FIT_INTERCEPT, 
                     random_state = RANDOM_SEED),
               ElasticNet(alpha = 0.1, l1_ratio = 0.5, 
                    max_iter=10000, tol=0.01, 
                    fit_intercept = SET_FIT_INTERCEPT, 
                    normalize = False, 
                    random_state = RANDOM_SEED),
               BaggingRegressor(DecisionTreeRegressor(random_state=RANDOM_SEED, max_features='log2'), 
                    n_estimators=100,max_samples=100, bootstrap=True, 
                    n_jobs=-1, random_state=RANDOM_SEED),
               RandomForestRegressor(n_estimators=100, max_leaf_nodes=12, bootstrap=True,
                    n_jobs=-1, random_state=RANDOM_SEED, max_features='log2'),
               AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), 
                    n_estimators=100, learning_rate=0.5, random_state=RANDOM_SEED),
               GradientBoostingRegressor(max_depth=5, n_estimators=100, 
                    learning_rate=1.0, random_state=RANDOM_SEED, max_features='log2'),
               GradientBoostingRegressor(max_depth=5, n_estimators=100, 
                    learning_rate=0.1, random_state=RANDOM_SEED, max_features='log2'),
               ExtraTreesRegressor(n_estimators=100, criterion='mse', max_depth=5, 
                    min_samples_split=2, min_samples_leaf=1, max_features='log2', 
                    bootstrap=True, random_state=RANDOM_SEED),
               BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
              ]
         
   


# let's evaulate using Strat Shuffle to preserve the ratios of the response
# variables in each fold of the test set
print ("\n\n\n************ Using Stratefied Shuffle Split ***********************")

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

def eval_model(data, labels):   

    # array to hold results
    cross_val_res1 = np.zeros((N_FOLDS, len(reg_methods)))
    cross_val_res2 = np.zeros((N_FOLDS, len(reg_methods)))
    r2_val_res = np.zeros((N_FOLDS, len(reg_methods)))
    exp_var_res = np.zeros((N_FOLDS, len(reg_methods)))
    
    # Ten-fold cross-validation with stratified sampling.
    stsp = StratifiedShuffleSplit(n_splits=N_FOLDS)
    fold_index = 0
    for train_index, test_index in stsp.split(data, labels):
       X_train, X_test = data.iloc[train_index], data.iloc[test_index]
       y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
       
       print ("\n\n***********************************************************************************\n") 
       print ("*************************************** FOLD = {} **********************************\n".format (fold_index))
       print ("************************************************************************************\n") 
         
       method_index = 0
       for model_name, method in zip(reg_methods, regress_list):
           print("\n\n\nRegression model evaluation for model:", model_name)
           print("Scikit_Learn method:", method)
           method.fit(X_train,y_train)         
           y_test_predict = method.predict(X_test)
            
           r2_val = r2_score(y_test, y_test_predict) 
           print("R-squared is:", r2_val)
           fold_method_res1 = mean_absolute_error(y_test, y_test_predict)
           fold_method_res2 = np.sqrt(mean_squared_error(y_test, y_test_predict))
           print(method.get_params(deep=True))
           print('Mean absolute error:', fold_method_res1)
           print('Root mean-squared error:', fold_method_res2)

           cross_val_res1[fold_index, method_index] = fold_method_res1
           cross_val_res2[fold_index, method_index] = fold_method_res2
           r2_val_res[fold_index, method_index] = r2_val    
          
           fpr, tpr, _ = roc_curve(y_test, y_test_predict)
           plt.figure(figsize=(8, 6))
           plt.title("ROC Curve")
           plot_roc_curve(fpr, tpr)
           plt.show()
                        
           method_index += 1
       fold_index += 1
       

    cross_val_res1_df = pd.DataFrame(cross_val_res1)
    cross_val_res1_df.columns = reg_methods
    
    cross_val_res2_df = pd.DataFrame(cross_val_res2)
    cross_val_res2_df.columns = reg_methods
    
    r2_val_res_df = pd.DataFrame(r2_val_res)
    r2_val_res_df.columns = reg_methods

    res1=cross_val_res1_df.mean()
    res2=cross_val_res2_df.mean()
    r2=r2_val_res_df.mean()
    
    tmp = pd.concat([res1, res2, r2], axis=1)   

    return tmp


#**************************************************
# Run the build and evaluate model loop
#**************************************************
print ("\n\n\n************ PER FOLD REGRESSION RESULTS  ")
orig_res = eval_model(train_data, target)
orig_res.columns = ['MAE', 'RMSE', 'R2']
sorted_res = orig_res.sort_values(by = 'MAE')



# Output results of cross-validation for comparison
#--------------------------------------------------
print ("\n\n\n************ AVERAGE OF REGRESSION RESULTS ACROSS ALL FOLDS ")
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      '\nMethod               Root mean-squared error', sep = '')     

print("Method\n{0}".format(sorted_res))

########################################################################
# get the feature importance for the model that did best
########################################################################
def get_importance(df, model_name):  
          
    col_names = df.columns
    feature_list = np.delete(col_names,0)
         
    
    X_train = df.iloc[:, 1:]
    y_train = df.iloc[:, 0]
           
    try:
        #for regress_list[model]:
        model_num = reg_methods.index(model_name)
        model = regress_list[model_num]
        
        model.fit(X_train,y_train)
        feature_import = np.round(model.feature_importances_,4)
        array_stack = np.column_stack([feature_list, feature_import])
        tmp_array = array_stack[np.argsort(array_stack[:, 1])]
        print('\n----------------------------------------------')
        print('Feature importance for method', model, '\n')
        print(np.array2string(tmp_array).replace('[[',' [').replace(']]',']'))
    except:
        print("**** !! Best method has no feature importance  !! ****", candidate)
       

print ("\n\n\n************ FEATURE IMPORTANCE ***********************")

# get model with lowest MAE and find feature importance
candidate = sorted_res.index[0]
get_importance(train, candidate)


