# Light gbm

# Importing important libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Insurance_Marketing-Customer-Value-Analysis.csv')

# Adding 'Effective to Date' variable
from dateutil.relativedelta import relativedelta
from datetime import date, datetime

dataset['Effective to date'] = pd.to_datetime(dataset['Effective To Date'])
today = pd.to_datetime('now')
dataset['diff_days'] = today - dataset['Effective to date']
dataset['diff_days']=dataset['diff_days']/np.timedelta64(1,'D')

v = [1,3,4,5,7,8,10,11,17,18,19,20,22,23]
v_not_categorical = [9,12,13,14,15,16,21,25]
X = dataset.iloc[:, v].values
y = dataset.iloc[:, 2].values
X_not_cat = dataset.iloc[:,v_not_categorical].values

# LABEL ENCODING the categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for i in range(len(v)):
    X[:,i] = labelencoder.fit_transform(X[:,i]).astype(np.float64)
    
X = X.astype(np.float64)

# Concatenating two categorical and non categorical columns
X= np.append(arr= X ,values = X_not_cat, axis =1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

import lightgbm as lgb
# create dataset for lightgbm
''''lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  '''

# train
gbm = lgb.LGBMRegressor(num_leaves=7,
                        learning_rate=0.05,
                        n_estimators=450,
                        boosting_type='gbdt',
                        max_depth = 6)
gbm.fit(X_train, y_train)


gbm.score(X_test,y_test)