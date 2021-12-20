# -*- coding: utf-8 -*-
"""
Created on Fri May 14 16:39:20 2021

@author: AlexandrosCMitronikas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, RobustScaler



#df_wPolicies['date'] = pd.to_datetime(df_wPolicies['date'])
df_wPolicies.head(10)

X = np.array(df_wPolicies)  # get input values
y = np.array(df_all['rr+1'])

#y = np.array(df_all['rr+5'])

X.shape
y.shape

#Normalization
min_max_scaler = MinMaxScaler().fit(X_test)
X_norm = min_max_scaler.transform(X)



#scaling data

scaler = StandardScaler().fit(X_train)
X_std = scaler.transform(X)

scaler = RobustScaler()
X = scaler.fit_transform(X.reshape(-1, 1))
y = scaler.fit_transform(y.reshape(-1, 1))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

#calling model and fitting
model = LinearRegression()

model.fit(X_train, y_train)

#making prediction
y_pred = model.predict(X_test)


# The intercept
print('Interccept: \n', model.intercept_)
# The coefficients
print('Coefficients: \n', model.coef_)

print('MSE: %.2f' % mean_squared_error(y_test, y_pred))
print('R^2: %.2f' % r2_score(y_test, y_pred))


#Setting parameters and options for grid search
parameters = [{
'fit_intercept': ['True', 'False'], 
'normalize': ['True', 'False'],
'copy_X': ['True', 'False'],
'n_jobs': ['None','-1']}]


grid = GridSearchCV(
        model(), parameters, scoring='accuracy'
    )
grid.fit(X_train, y_train)


#Getting model result 
r_sq = model.score(X, y)
print('coefficient of determination:', r_sq)


 
