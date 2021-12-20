# -*- coding: utf-8 -*-
"""
Created on Fri May 14 21:11:37 2021

@author: AlexandrosCMitronikas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from sklearn.model_selection import GridSearchCV


#After feature selection 
#for rr+1 
X=np.array(x_ForRr1)
y=np.array(df_all['rr+1'])
#for rr+5 
X=np.array(x_ForRr5)
y=np.array(df_all['rr+5'])

X.shape
y.shape



#Scaling
# Rescale data (between 0 and 1)
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()  
X = sc_X.fit_transform(X) 


#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=25)


from sklearn.svm import SVR

regressor = SVR()
regressor.fit(X_train,y_train)
regressor.score(X_test,y_test)


#Grid Search
param_grid = { 'C':[0.1,1,10,100,1000],'kernel':['poly','rbf','sigmoid', 'linear'],'degree':[1,2,3,4,5,6],'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVR(),param_grid)
grid.fit(X_train,y_train.ravel())

#Grid Search
param_grid = { 'C':[10,100],'kernel':['linear'],'degree':[1,2,3,4,5,6],'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVR(),param_grid)
grid.fit(X_train,y_train)

print(grid.best_params_)
print(grid.score(X_test,y_test))



# FIT THE MODEL USING BEST ESTIMATOR

model = SVR(C=1000,  degree=1, epsilon=0.1, gamma=0.0001,
  kernel='linear')

model.fit(X_train,y_train.ravel())
results = model.predict(X_test)
print(results)
model.score(X_test,y_test)



# Visualising the SVR results 
plt.scatter(X, y, color = 'red') 
plt.plot(X, regressor.predict(X), color = 'blue') 
plt.xlabel('Position level') 
plt.ylabel('Salary') 
plt.show()



#Feature scale comparison
import matplotlib.pyplot as plt
import numpy as np

names = df_wPolicies.columns

def featureBoxPlot(X, names):
    # change figure size
    plt.rcParams["figure.figsize"] = [7, 3]
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Feature Scale Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(X, showmeans=True)
    ax.set_xticklabels(names, fontsize=8.5)
    plt.show()
    return

featureBoxPlot(df_wPolicies.values, names)
