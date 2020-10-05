# Building random forest regression model

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

# Fitting random forest regressor model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300 ,random_state = 0)
regressor.fit(X_train,y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

# Checking r^2 value
from sklearn.metrics import r2_score
r2_score(y_test ,y_pred)


'''
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
rfc = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight='balanced')
boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)
boruta_selector.fit(X_train,y_train)

print(“==============BORUTA==============”)
print (boruta_selector.n_features_)


from sklearn.metrics import r2_score
from rfpimp import permutation_importances

def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))

perm_imp_rfpimp = permutation_importances(rf, X_train, y_train, r2)   '''