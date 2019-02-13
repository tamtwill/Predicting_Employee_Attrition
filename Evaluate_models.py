#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 11:18:12 2019

@author: Tamara Williams
"""


import pandas as pd
import numpy as np



from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor,\
 AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score

from matplotlib import pyplot as plt


RND_ST = np.random.RandomState(31)

# initialize various system variables
RANDOM_SEED = 13
SET_FIT_INTERCEPT=True

# set the number of folds for cross-validation
N_FOLDS = 10


datapath = '~/gitRepo/Predicting_Employee_Attrition/data/'
outpath = '~/gitRepo/Predicting_Employee_Attrition/output/'

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

train = pd.read_csv(datapath+'processed_train.csv', delimiter=',', encoding='utf-8')
cols = train.columns.tolist()
cols.insert(0, cols.pop(cols.index('has_quit')))
train = train.loc[:, cols]

train_data = train.copy(deep=True)
train_data = train_data.drop(columns=['has_quit'])
target = train['has_quit']


# Build and cross-validate regression models
#--------------------------------------------------
#----- Setup the list of models to look at

reg_methods = ['Logisitc_Regression', 'Ridge_Regression', 'Lasso_Regression', 
          'ElasticNet_Regression', 'Bagging Ensemble', 
          'Random Forest', 'AdaBoost','Gradient Boost 1.0','Gradient Boost .1', 
          'Extra Trees']

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
                    bootstrap=True, random_state=RANDOM_SEED)
              ]
         
   

def eval_model(df):

    # array to hold results
    cross_val_res = np.zeros((N_FOLDS, len(reg_methods)))
    r2_val_res = np.zeros((N_FOLDS, len(reg_methods)))
        
    k_folds = KFold(n_splits=N_FOLDS, shuffle=False, random_state=RANDOM_SEED)

    fold_index = 0
    for train_index, test_index in k_folds.split(df):
        print("Index is:", fold_index,"________________")
        # set up the split between train and test
        # relies on response being in df[0]
        X_train = df.iloc[train_index, 1:df.shape[1]]
        X_test = df.iloc[test_index, 1:df.shape[1]]
        y_train = df.iloc[train_index, 0]
        y_test = df.iloc[test_index, 0]   
        print('Shape of input data for this fold:','Data Set: (Observations, Variables)')
        print('X_train:', X_train.shape)
        print('X_test:',X_test.shape)
        print('y_train:', y_train.shape)
        print('y_test:',y_test.shape)
    
        
        method_index = 0
        for model_name, method in zip(reg_methods, regress_list):
            print("\n\n\nRegression model evaluation for model:", model_name)
            print("Scikit_Learn method:", method)
            method.fit(X_train,y_train)
            
            # run the eval on this fold
            y_test_predict = method.predict(X_test)
            r2_val = r2_score(y_test, y_test_predict) 
            print("R-squared is:", r2_val)
            fold_method_res = np.sqrt(mean_squared_error(y_test, y_test_predict))
            print(method.get_params(deep=True))
            print('Root mean-squared error:', fold_method_res)
            cross_val_res[fold_index, method_index] = fold_method_res
            r2_val_res[fold_index, method_index] = r2_val       
           
            fpr, tpr, _ = roc_curve(y_test, y_test_predict)
            plt.figure(figsize=(8, 6))
            plt.title("ROC Curve")
            plot_roc_curve(fpr, tpr)
            plt.show()
                        
            method_index += 1
      
        fold_index += 1
    
    cross_val_res_df = pd.DataFrame(cross_val_res)
    cross_val_res_df.columns = reg_methods
    r2_val_res_df = pd.DataFrame(r2_val_res)
    r2_val_res_df.columns = reg_methods

    res=cross_val_res_df.mean()
    r2=r2_val_res_df.mean()
    
    tmp = pd.concat([res, r2], axis=1)   

    return tmp
    print ("***********************************")


#**************************************************
# Run the build and evaluate model loop
#**************************************************
# do evaluation with original data
print ("\n\n\n************ SUMMARY REGRESSION RESULTS USING RAW DATA ")
orig_res = eval_model(train)
orig_res.columns = ['RMSE', 'R2']
sorted_res = orig_res.sort_values(by = 'RMSE')



# Output results of cross-validation for comparison
#--------------------------------------------------
print ("\n\n\n************ SUMMARY REGRESSION RESULTS USING RAW DATA ")
print('Average results from ', N_FOLDS, '-fold cross-validation\n',
      '\nMethod               Root mean-squared error', sep = '')     

print("Method\n{0}".format(sorted_res))

########################################################################
# get the feature importance for the models that have it
# skip Gradient Boost 1.0, it is not successful in the RMSE value
# so it is not needed here
########################################################################
def get_importance(df):  

   
    reg_methods = ['Random Forest', 'AdaBoost','Gradient Boost .1','Extra Trees']
    
    regress_list = [RandomForestRegressor(n_estimators=100, max_leaf_nodes=12, 
                    n_jobs=-1, random_state=RANDOM_SEED, max_features='log2'),
               AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), 
                    n_estimators=100, learning_rate=0.5, 
                    random_state=RANDOM_SEED),
               GradientBoostingRegressor(max_depth=5, n_estimators=100, 
                    learning_rate=0.1, random_state=RANDOM_SEED, max_features='log2'),
               ExtraTreesRegressor(n_estimators=100, criterion='mse', max_depth=5, 
                    min_samples_split=2, min_samples_leaf=1, max_features='log2', 
                    bootstrap=True, random_state=RANDOM_SEED)]
           
    col_names = df.columns
    feature_list = np.delete(col_names,0)
         
    
    X_train = df.iloc[:, 1:]
    y_train = df.iloc[:, 0]
           
    method_index = 0
    for model_name, method in zip(reg_methods, regress_list):
       method.fit(X_train,y_train)
       feature_import = np.round(method.feature_importances_,4)
       array_stack = np.column_stack([feature_list, feature_import])
       tmp_array = array_stack[np.argsort(array_stack[:, 1])]
       print('\n----------------------------------------------')
       print('Feature importance for method', model_name, '\n')
       print(np.array2string(tmp_array).replace('[[',' [').replace(']]',']'))
       method_index += 1
       

print ("\n\n\n************ FEATURE IMPORTANCE RAW DATA ***********************")

get_importance(train)

