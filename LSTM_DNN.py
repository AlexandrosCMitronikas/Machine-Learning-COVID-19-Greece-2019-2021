# -*- coding: utf-8 -*-
"""
Created on Mon May 17 02:48:36 2021

@author: AlexandrosCMitronikas
"""

from google.colab import drive
#drive.mount('/content/drive')

import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import *

import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.preprocessing import StandardScaler, RobustScaler

import tensorflow as tf


## Split function
def split_func(df):
  #df_new.loc[:,'new_rr'] = df_new.loc[:,'Reproduction rate'].shift(-1)
  #df.drop(df.tail(1).index, inplace=True)
  X_train, X_test = df_noPolicies[0:275].values, df_noPolicies[276:].values
  y_train, y_test = df_all['rr+1'][0:275].values, df_all['rr+1'][276:].values
  return X_train, y_train, X_test, y_test

## Prep-processing function
def autoregressive_pre(X, y):
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))
    return X, y

def plot_results(name, history):
    fig = plt.figure(figsize = (20, 5))
    ax = fig.add_subplot(131)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_ylabel('mse')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'val'], loc='upper left')

    ax = fig.add_subplot(132)
    ax.plot(history.history['mean_absolute_error'])
    ax.plot(history.history['val_mean_absolute_error'])
    ax.set_ylabel('mae')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'val'], loc='upper left')

    plt.show()
    
    #model.add(Dense(Hyperparameters['Dense1'][i]], activation='relu'))
hyperparameters={'LSTM':[1,10,100,200,300,500,700,1000]}
hyperparameters['LSTM'][1]



def lstm(input_shape):
  
  # create and fit the LSTM network
  model = Sequential()
  model.add(BatchNormalization(input_shape=(1, input_shape)))
  model.add(LSTM(128))
  model.add(Dense(1))

  return model



from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model



def conv_1d(input_shape):
  model = Sequential()
  model.add(BatchNormalization(input_shape=(input_shape, 1)))
  model.add(Conv1D(64, 2, strides=2, activation="relu"))
  model.add(BatchNormalization())
  model.add(Conv1D(128, 2, activation="relu"))
  model.add(BatchNormalization())
  model.add(GlobalAveragePooling1D())
  model.add(Dense(128, activation="relu"))
  model.add(BatchNormalization())
  model.add(Dense(1))

  return model



def DNN(input_shape):
  
  model = Sequential()
  
  # The Input Layer : + Hyperparameters['Dense1']
  model.add(Dense(128,  activation='relu', input_dim = input_shape))

  # The Hidden Layers :
  model.add(Dense(60, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(400, activation='relu'))

  # The Output Layer :
  model.add(Dense(1, activation='linear'))

  return model





def run_exps_DL(df):
    X_train, y_train, X_test, y_test = split_func(df)
    
    X_train, y_train = autoregressive_pre(X_train, y_train)
    X_test, y_test = autoregressive_pre(X_test, y_test)

    input_shape = X_train.shape[1]

    models = [('Deep_Neural_network', DNN(input_shape))]

  #('lstm', lstm(input_shape))
  #('Conv1D', conv_1d(input_shape)),
  #('Deep_Neural_network', DNN(input_shape)),

    scoring = ['model', 'mse', 'r_sq', 'mae', 'time(s)']
    this_df = pd.DataFrame(columns=scoring)
    i = 0

    for name, model in models:

      if name=='lstm':
        X_train = X_train.reshape(-1, 1, input_shape)
        X_test  = X_test.reshape(-1, 1, input_shape)
        y_train = y_train.reshape(-1, 1, 1)
        y_test = y_test.reshape(-1, 1, 1)
      elif name == 'Conv1D':
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
      else:
        X_train = X_train.reshape(-1, input_shape)
        X_test = X_test.reshape(-1, input_shape)

      start = datetime.now()
      this_df.loc[i, 'model'] = name # appending name of model

      model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

      print(model.summary())

      es = EarlyStopping(monitor='val_mean_absolute_error', mode='min', verbose=0, patience=20)
      mc = ModelCheckpoint(str(name) + '.h5', monitor='val_mean_absolute_error', 
                          mode='min', verbose=0, save_best_only=True)
      
      history = model.fit(X_train, y_train, batch_size = 2, 
                          epochs=500, validation_data=(X_test, y_test),
                          callbacks=[es, mc], verbose = 0) # Fitaroume to montelo

      ## getting evaluation metrics
      y_pred = model.predict(X_test)
      
      y_test = y_test.reshape(-1,1)
      y_pred = y_pred.reshape(-1,1)

      this_df.loc[i, 'mse'] = mean_squared_error(y_test, y_pred)
      this_df.loc[i, 'r_sq'] = mean_absolute_error(y_test, y_pred)
      this_df.loc[i, 'mae'] = r2_score(y_test, y_pred)

      end = datetime.now()
      this_df.loc[i, 'time(s)'] = (end-start).total_seconds()

      i+=1
      print(this_df)
      plot_results(name, history)

    return this_df


#EARLY stoppings
run_exps_DL(df_wPolicies)


