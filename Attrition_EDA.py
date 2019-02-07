#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:54:36 2019

Data Source: 
https://www.ibm.com/communities/analytics/watson-analytics-blog/hr-employee-attrition/

@author: Tamara Williams
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import gc


rnd_st = np.random.RandomState(31)

## see  https://github.com/numpy/numpy/issues/11411 and 
## https://stackoverflow.com/questions/53334421/futurewarning-with-distplot-in-seaborn 
## for why FutureWarnigns are suppressed
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 

# set a pretty default palette
sns.palplot(sns.husl_palette(10, h=.5))
col_list_palette = sns.husl_palette(10, h=.5)
sns.set_palette(col_list_palette)



datapath = '~/gitRepo/Predicting_Employee_Attrition/data/'
outpath = '~/gitRepo/Predicting_Employee_Attrition/output/'

train = pd.read_csv(datapath+'WA_Fn-UseC_-HR-Employee-Attrition.csv', delimiter=',', encoding='utf-8')

# look at general aspects of the data
print(train.shape,'\n')
print(train.head(5),'\n')
print(train.info(), '\n')
print(train.dtypes, '\n')

print(train.describe(),'\n')

# explicitly check for missing values
train.isnull().values.any()
train.isna().sum()

# drop columns which are not useful
# everyone is over 18, works 80 hours and EmployeeCount is always 1
train = train.drop( columns = ['Over18', 'EmployeeCount', 'StandardHours'], axis = 1)

# not wildy useful given the number of features
#sns.pairplot(train)

# let's look at some of the distributions of the data
# start by looking at the catagorical data
plot_number = 1
df_objects = train.select_dtypes([object])
plt.figure(figsize=(12, 16), facecolor='white')
for c in df_objects.columns:
    col_counts = train[c].value_counts()
    col_counts.sort_index(inplace=True)
    ax = plt.subplot(4, 2, plot_number)
    sns.barplot(x = col_counts.index, y = col_counts)
    str_title = "counts by " + str(c)
    plt.ylabel(str_title)
    plt.xticks(rotation=45)
    plot_number = plot_number+1
plt.tight_layout()
plt.show()

print (train.Attrition.value_counts())
print('Attrition rate = ', 100*(train.Attrition.value_counts()['Yes']/train.Attrition.value_counts()['No']))

    
# now the numeric data    
plot_number = 1
plt.figure(figsize=(12, 16), facecolor='white')
df_ints = train.select_dtypes([int])
for c in df_ints.columns:
    ax = plt.subplot(8, 3, plot_number)
    sns.distplot(train[c])
    plt.xticks(rotation=45)
    plot_number = plot_number+1
plt.tight_layout()
plt.show()
    
    
disc_list = ['Age', 'Attrition', 'BusinessTravel', 'Department',
       'DistanceFromHome', 'Education', 'EducationField', 
       'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
       'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
       'NumCompaniesWorked', 'OverTime','PerformanceRating', 
       'RelationshipSatisfaction','StockOptionLevel', 'TotalWorkingYears', 
       'TrainingTimesLastYear','WorkLifeBalance', 'YearsAtCompany', 
       'YearsInCurrentRole','YearsSinceLastPromotion', 'YearsWithCurrManager']
    
## Alternate  
plot_number = 1
plt.figure(figsize=(12, 60), facecolor='white')
for c in disc_list:
    ax = plt.subplot(16, 2, plot_number)
    sns.countplot(x=train[c], data=train)
    plt.xticks(rotation=90)
    plot_number = plot_number+1
plt.tight_layout()
plt.show()


# let's look at some of the relationships in the data
fig, ((ax1, ax2, ax3)) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10,5))
sns.stripplot(x="PerformanceRating", y="Attrition", data=train, jitter=True, ax = ax1)
sns.stripplot(x="JobSatisfaction", y="Attrition", data=train, jitter=True, ax = ax2)
sns.stripplot(x="WorkLifeBalance", y="Attrition", data=train, jitter=True, ax = ax3)
plt.tight_layout()

fig, ((ax4, ax5)) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10,5))
sns.stripplot(x="Age", y="Attrition", data=train, jitter=True, ax = ax4)
sns.stripplot(x="DistanceFromHome", y="Attrition", data=train, jitter=True, ax = ax5)
plt.tight_layout()

density_pairs = [['JobSatisfaction','DistanceFromHome'], ['JobSatisfaction','NumCompaniesWorked'],['JobSatisfaction','MonthlyIncome'],
                 ['JobSatisfaction','WorkLifeBalance'],['JobSatisfaction','YearsSinceLastPromotion'], ['MonthlyRate','MonthlyIncome'],
                 ['PercentSalaryHike', 'MonthlyIncome'], ['TotalWorkingYears', 'MonthlyIncome'], ['TotalWorkingYears', 'YearsAtCompany']]

plot_number = 1
plt.figure(figsize=(10, 10), facecolor='white')
for p in density_pairs:
    x = p[0]
    y = p[1]
    ax = plt.subplot(3, 3, plot_number)
    sns.kdeplot(train[x], train[y], cmap="Blues", shade=True, shade_lowest=False)
    plot_number = plot_number+1
plt.tight_layout()

df_money = train[['MonthlyIncome', 'MonthlyRate']].copy()
f, ax = plt.subplots(figsize=(3,4))
sns.boxplot(x="variable", y="value", data=pd.melt(df_money))
plt.show()

df_service = train[['TotalWorkingYears', 'NumCompaniesWorked', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsInCurrentRole', 'YearsWithCurrManager' ]].copy()
f, ax = plt.subplots(figsize=(3,4))
sns.boxplot(x="variable", y="value", data=pd.melt(df_service))
plt.xticks(rotation=60)
plt.show()

df_sat = train[['EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'RelationshipSatisfaction']].copy()
f, ax = plt.subplots(figsize=(3,4))
sns.boxplot(x="variable", y="value", data=pd.melt(df_sat))
plt.xticks(rotation=60)
plt.show()

df_other = train[['DistanceFromHome', 'PercentSalaryHike']].copy()
f, ax = plt.subplots(figsize=(3,4))
sns.boxplot(x="variable", y="value", data=pd.melt(df_other))
plt.xticks(rotation=60)
plt.show()

# looking at a correlation matrix
t_corr = train.corr()

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.zeros_like(t_corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(t_corr, mask=mask, vmax=.3, square=True, center = 0, cmap=sns.color_palette("BrBG", 7))
plt.show()

print(t_corr)

## plotting the pair plots  - takes too long to run and doesn't add much to the understanding of the data
#sns.pairplot(coded_train)
#plt.show()



#del df_money
#del df_service
#del df_sat
#del df_service
gc.collect()


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

tmp = train.copy(deep = True)
tmp = tmp.drop(['Attrition', 'BusinessTravel', 'Department',
       'EducationField', 'Gender', 'JobRole', 'MaritalStatus',
       'OverTime'], axis = 1)
X = add_constant(tmp)
VIF = pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)
print(VIF)
## JobLevel and MonthlyIncome have a high degree of collinearity - may 
## want to take one out



# let's create numeric representations of the yes/no binary columns
# use descriptive column names to show how the translation was done
train['has_quit'] = train['Attrition'].map({'Yes':1, 'No':0})
train['works_OT'] = train['OverTime'].map({'Yes':1, 'No':0})
train['is_male'] = train['Gender'].map({'Male':1, 'Female':0})
train['is_divorced'] = train['MaritalStatus'].map({'Divorced':1, 'Married':0, 'Single':0})
train['is_married'] = train['MaritalStatus'].map({'Divorced':0, 'Married':1, 'Single':0})
train['is_single'] = train['MaritalStatus'].map({'Divorced':0, 'Married':0, 'Single':1})
train['travels'] = train['BusinessTravel'].map({'Non-Travel':0, 'Travel_Rarely':1, 'Travel_Frequently':2})

# now for the catagories where one-hot encoding is needed
tmp = pd.get_dummies(train['JobRole'])
tmp.columns = ['is_healthcare_rep', 'is_HR', 'is_lab_tech', 'is_mgr', 'is_mfg_dir', 'is_res_dir', 'is_res_sci', 'is_sales_exc', 'is_sales_rep'] 
train = train.join(tmp)

tmp = pd.get_dummies(train['Department'])
tmp.columns = ['in_HR', 'in_R&D', 'in_sales'] 
train = train.join(tmp)

tmp = pd.get_dummies(train['EducationField'])
tmp.columns = ['edu_HR', 'edu_life_sci', 'edu_marketing', 'edu_med', 'edu_other', 'edu_tech'] 
train = train.join(tmp)

coded_train = train.copy(deep=True)

#del train
gc.collect()

coded_train = coded_train.drop(['Attrition', 'BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime'], axis = 1)

# save for later use
coded_train.to_csv(datapath+'processed_train.csv', sep=',', date_format = 'string', index = False, encoding = 'utf-8')

    

