# -*- coding: utf-8 -*-
"""
Created on Fri May 14 00:16:03 2021

@author: alexc
"""

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler, RobustScaler

features = df_noPolicies
#setting X including policy variables
#features = df_wPolicies

# Labels are the values we want to predict
labels = np.array(df_all['rr+1'])
#setting y for a five day prediction
#labels = np.array(df_all['rr+5'])

# Convert to numpy array
features = np.array(features)

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 69)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 69, max_depth=30, max_features=20)
# Train the model on training data
rf.fit(train_features, train_labels);


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')




# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), '.')
