# Building regression tree model

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
X = pd.DataFrame(dataset.iloc[:, v])
y = dataset.iloc[:, 2].values
X_not_cat = dataset.iloc[:,v_not_categorical].values

# Encoding categorical data
X = pd.get_dummies(X, drop_first=True).values
# Concatenating two categorical and non categorical variables
X= np.append(arr= X ,values = X_not_cat, axis =1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting the Decision tree Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor =  DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train,y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

# Checking R values
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
