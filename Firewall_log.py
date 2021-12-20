# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:13:23 2021

@author: gkl_a
"""

import pandas as pd
import numpy as np
import sklearn
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from pandas import read_csv
from pandas import read_xlx
from sklearn import preprocessing
from sklearn import datasets, linear_model
from utilities import visualize_classifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor



# visualisation
%matplotlib inline
sns.set(color_codes=True)

# To load the data
df = pd.read_csv("log2.csv")

# To display the top 5 rows
df.head(5)

# To display the bottom 5 rows
df.tail(5)

# Checking the data type
df.dtypes

# Total number of rows and columns
df.shape

# Rows containing duplicate data
duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ",
      duplicate_rows_df.shape)

# Used to count the number of rows before removing the data
df.count()

# Drop Duplicates
df.drop_duplicates(inplace=True)

# Finding the null values.
print(df.isnull().sum())

df["Action"] = df["Action"].astype('category')
df["Action_codes"] = df["Action"].cat.codes
df["Action"] = df["Action"].cat.codes
df.dtypes

# Detecting the Outliers
sns.boxplot(x=df['Source Port'])
sns.boxplot(x=df['Destination Port'])
sns.boxplot(x=df['NAT Source Port'])
sns.boxplot(x=df['NAT Destination Port'])
sns.boxplot(x=df['Action_codes'])
sns.boxplot(x=df['Bytes'])
sns.boxplot(x=df['Packets'])
sns.boxplot(x=df['Elapsed Time (sec)'])

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

df = df[~((df < (Q1-1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape
df.describe()

# Plotting a Histogram
df.Action.value_counts().nlargest(40).plot(kind='bar', figsize=(10, 5))
plt.title('Output')
plt.ylabel('NAT Destination Port')
plt.xlabel('Action_codes')

# Plotting a scatter plot
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(df['NAT Destination Port'], df['Action_codes'])
plt.title('Scatter plot between Action_codes and NAT Destination Port')
ax.set_xlabel('Action_codes')
ax.set_ylabel('NAT Destination Port')
plt.show()

X = df.drop(['Action', 'Action_codes'], axis=1)
y = df['Action_codes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,
                                                    random_state=17)

# Binarize data
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(df)
print("\nBinarized data:\n", data_binarized)

# Print mean and standard deviation
print("\nBEFORE:")
print("Mean =", y_train.mean(axis=0))
print("Std deviation =", y_train.std(axis=0))

# Remove mean
X_train_scaled = preprocessing.scale(X_train)
y_train_scaled = preprocessing.scale(y_train)
X_test_scaled = preprocessing.scale(X_test)
y_test_scaled = preprocessing.scale(y_test)
print("\nAFTER:")
print("Mean =", y_train_scaled.mean(axis=0))
print("Std deviation =", y_train_scaled.std(axis=0))

# Min max scaling
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(df)
print("\nMin max scaled data:\n", data_scaled_minmax)

# Normalize data
data_normalized_l1 = preprocessing.normalize(df, norm='l1')
data_normalized_l2 = preprocessing.normalize(df, norm='l2')
print("\nL1 normalized data:\n", data_normalized_l1)
print("\nL2 normalized data:\n", data_normalized_l2)


# Finding the relations between the variables.
plt.figure(figsize=(10, 5))
c = df.corr()
sns.heatmap(c, cmap="BrBG", annot=True, fmt=".2f")
c

# Dropping irrelevant columns
df = df.drop(['Bytes', 'Bytes Sent', 'Bytes Received', 'Packets', 'pkts_sent', 'pkts_received'],
             axis=1)
df.head(5)

# Plot them
list(set(df.dtypes.tolist()))
df_num = df.select_dtypes(include=['int64', 'int8'])
df_num.head()
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
df_num_corr = df_num.corr()['Action_codes'][:-1]
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with Action_codes:\n{}".format(len(golden_features_list), golden_features_list))

# K Neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train)
y_predicted = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_predicted)
knn_f1 = f1_score(y_test, y_predicted, average='weighted')
print('Accuracy (KNN): ', "%.2f" % (knn_accuracy*100))
print('F1 (KNN): ', "%.2f" % (knn_f1*100))
print('KNN score: %.f' % knn.fit(X_train, y_train).score(X_test, y_test))

# Logistic Regression
logistic = linear_model.LogisticRegression(max_iter=8000).fit(X_train, y_train)
y_predicted = logistic.predict(X_test)
logistic_accuracy = accuracy_score(y_test, y_predicted)
logistic_f1 = f1_score(y_test, y_predicted, average='weighted')
print('Accuracy (Logistic): ', "%.2f" % (logistic_accuracy*100))
print('F1 (Logistic): ', "%.2f" % (logistic_f1*100))

# Linear Kernel
lnr = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)
y_predicted = lnr.predict(X_test)
lnr_accuracy = accuracy_score(y_test, y_predicted)
lnr_f1 = f1_score(y_test, y_predicted, average='weighted')
print('Accuracy (Linear Kernel): ', "%.2f" % (lnr_accuracy*100))
print('F1 (Linear Kernel): ', "%.2f" % (lnr_f1*100))
print('Classification report for', "%s" % lnr)
print(metrics.classification_report(y_test, y_predicted))
print(classification_report(y_test, y_predicted))
print('Confusion matrix')
print(metrics.confusion_matrix(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))

# Create confusion matrix
confusion_mat = confusion_matrix(y_test, y_predicted)

# Visualize confusion matrix
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange(4)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.show()

# Polyonomial Kernel
poly = svm.SVC(kernel='poly', degree=8, C=0.01).fit(X_train, y_train)
y_predicted = poly.predict(X_test)
poly_accuracy = accuracy_score(y_test, y_predicted)
poly_f1 = f1_score(y_test, y_predicted, average='weighted')
print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))
print('Classification report for' "%s" % poly)
print('Confusion matrix')
print(metrics.classification_report(y_test, y_predicted))
print(metrics.confusion_matrix(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))
print(classification_report(y_test, y_predicted))

# Gausian Kernel
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.01).fit(X_train, y_train)
y_predicted = rbf.predict(X_test)
rbf_accuracy = accuracy_score(y_test, y_predicted)
rbf_f1 = f1_score(y_test, y_predicted, average='weighted')
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))
print('Classification report for' "%s" % rbf)
print('Confusion matrix')
print(metrics.classification_report(y_test, y_predicted))
print(metrics.confusion_matrix(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))
print(classification_report(y_test, y_predicted))

# Sigmoid Kernel
sigm = svm.SVC(kernel='sigmoid', gamma=0.7, C=0.01).fit(X_train, y_train)
y_predicted = sigm.predict(X_test)
sigm_accuracy = accuracy_score(y_test, y_predicted)
sigm_f1 = f1_score(y_test, y_predicted, average='weighted')
print('Accuracy (Sigmoid Kernel): ', "%.2f" % (sigm_accuracy*100))
print('F1 (Sigmoid Kernel): ', "%.2f" % (sigm_f1*100))
print('Classification report for' "%s" % sigm)
print('Confusion matrix')
print(metrics.classification_report(y_test, y_predicted))
print(metrics.confusion_matrix(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))
print(classification_report(y_test, y_predicted))

print(df.info())
df.head(3)
