#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 07:51:10 2025

@author: dliu
"""

import tensorflow
import pandas as pd
import keras
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn
import h5py


#### SETUP/SPLIT TRAINING/VALIDATION DATASETS ####################################

# Reading dataset
dataset = pd.read_csv() # insert whole data table

# Predictor dataset
predictors = dataset[] # data table containing the target column
# target dataset
targets = dataset[] # data table containing the target column

# Split predictors and targets into training and testing datasets
pred_train, pred_test, targ_train, targ_test = train_test_split(predictors,targets,test_size= 0.2,random_state=0)




#### MAKING THE MODEL #######################################################

# Depending on size of dataset, may consider reducing nodes in layers to avoid overfitting. 
# Original model was used for training dataset with 3049 variables.

# Initialize network constructor
model = keras.models.Sequential()

# Add an input layer
model.add(keras.layers.Dense(3050, activation='relu', input_shape=(pred_train.shape[1],))) 

# Add first hidden layer
model.add(keras.layers.Dense(2000, activation='relu'))

# Add second hidden layer
model.add(keras.layers.Dense(2000, activation='relu'))

# Add output layer
model.add(keras.layers.Dense(1,activation='sigmoid'))

# Model specs and features
print(model.summary())
print(model.get_config())
print(model.get_weights())

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

n = 145                 
model.fit(pred_train, targ_train,epochs=n, batch_size=10, verbose=1)

model.save('RARExDrug.h5')



#### ASSESS MODEL PERFORMANCE  ##############################################

# Predict scores
predictions = model.predict(pred_test)

score = model.evaluate(pred_test,targ_test,verbose=1)

# Confusion matrix
print('Confusion matrix:\n')
print(sklearn.metrics.confusion_matrix(targ_test, predictions))
       
# Precision 
print('Precision')
print(sklearn.metrics.precision_score(targ_test, predictions))

# Recall
print('Recall')
print(sklearn.metrics.recall_score(targ_test, predictions))

# F1 score
print('F1 score')
print(sklearn.metrics.f1_score(targ_test,predictions))

# Cohen's kappa
print("Cohen's kappa")
print(sklearn.metrics.cohen_kappa_score(targ_test, predictions))












