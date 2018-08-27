#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:20:10 2017

@author: Krist Papadopoulos

Model: Calibrated Random Tiny Lattices
Hyperparameters: L1/L2 Regularization
Dataset: Boston Housing Dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
import tensorflow_lattice as tfl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import itertools
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold


def get_train_input_fn(train_x, train_y, titles, batch_size, num_epochs=None, shuffle=False):
    
  x_train={k:train_x[:,v] for k,v in zip(titles,range(train_x.shape[1]))}

  return  tf.estimator.inputs.numpy_input_fn(
          x=x_train,
          y=train_y,
          batch_size=batch_size,
          shuffle=shuffle,
          num_epochs=num_epochs,
          num_threads=1)   
    
def get_test_input_fn(test_x, test_y, titles, num_epochs, shuffle=False):
        
  x_test={k:test_x[:,v] for k,v in zip(titles,range(test_x.shape[1]))}
        
  return  tf.estimator.inputs.numpy_input_fn(
          x=x_test,
          y=test_y,
          shuffle=shuffle,
          num_epochs=num_epochs,
          num_threads=1)         


def create_feature_columns():
  """Creates feature columns for Boston Data."""

  # No categorical features 

  # Numerical (continuous) base columns.
  crime_rate =      tf.feature_column.numeric_column("CRIM")
  lot_size =        tf.feature_column.numeric_column("ZN")
  non_retail =      tf.feature_column.numeric_column("INDUS")
  charles_river =   tf.feature_column.numeric_column("CHAS")
  nitrous_ox =      tf.feature_column.numeric_column("NOX")
  avg_rooms =       tf.feature_column.numeric_column("RM")
  home_age =        tf.feature_column.numeric_column("AGE")
  employment_dis =  tf.feature_column.numeric_column("DIS")
  highways =        tf.feature_column.numeric_column("RAD")
  tax_rate =        tf.feature_column.numeric_column("TAX")
  pupil_teacher =   tf.feature_column.numeric_column("PTRATIO")
  black =           tf.feature_column.numeric_column("B")
  lower_status =    tf.feature_column.numeric_column("LSTAT")

  return [crime_rate,lot_size,non_retail,charles_river,nitrous_ox,avg_rooms,home_age,employment_dis,highways,tax_rate,pupil_teacher,black,lower_status]


def create_quantiles(quantiles_dir, train_x, train_y, titles):
  """Creates quantiles directory if it doesn't yet exist."""
  # Reads until input is exhausted, 10000 at a time.
  tfl.save_quantiles_for_keypoints(
          input_fn=get_train_input_fn(train_x, train_y, titles, batch_size=train_x.shape[0],),
          save_dir=quantiles_dir,
          feature_columns=create_feature_columns(),
          num_steps=1)
  
  
def _pprint_hparams(hparams):
  """Pretty-print hparams."""
  print("* hparams=[")
  for (key, value) in sorted(six.iteritems(hparams.values())):
    print("\t{}={}".format(key, value))
  print("]")
      

def create_calibrated_rtl(feature_columns, config, quantiles_dir):
  feature_names = [fc.name for fc in feature_columns]
  hparams = tfl.CalibratedRtlHParams(
      feature_names=feature_names,
      num_keypoints=10,
      learning_rate=0.1,
      lattice_size=2,
      lattice_rank=4,
      num_lattices=100,
      calibration_l2_reg=0.01)
  # Specific feature parameters.
  #hparams.set_feature_param("capital_gain", "lattice_size", 8)
  #hparams.set_feature_param("native_country", "lattice_size", 8)
  #hparams.set_feature_param("marital_status", "lattice_size", 4)
  #hparams.set_feature_param("age", "lattice_size", 8)

  _pprint_hparams(hparams)
  return tfl.calibrated_rtl_regressor(
      feature_columns=feature_columns,
      model_dir=config.model_dir,
      config=config,
      hparams=hparams,
      quantiles_dir=quantiles_dir)
  

def create_estimator(config, quantiles_dir):
  feature_columns = create_feature_columns()
  return create_calibrated_rtl(feature_columns, config, quantiles_dir) 


def train(estimator, train_x, train_y, titles):
  """Trains estimator and optionally intermediary evaluations."""
  return estimator.train(input_fn=get_train_input_fn(train_x, train_y, titles, batch_size=100), steps=5000) 


def evaluate(estimator, test_x, test_y, titles):
  """Evaluates and prints results"""
  evaluation = estimator.evaluate(input_fn=get_test_input_fn(test_x, test_y, titles, 1), steps=1)  
  print(evaluation)


def predict(estimator, test_x, test_y, titles):
  """Predicts and save new dta points"""
  
  prediction = estimator.predict(input_fn=get_test_input_fn(test_x, test_y, titles, 1,))
    
  y_predict = np.stack(list((p["predictions"] for p in itertools.islice(prediction, test_x.shape[0])))).flatten()
  
  return y_predict
  

if __name__ == "__main__":
    
  data = load_boston()
  
  X = data.data
  y = data.target
  titles = data.feature_names

  # Prepare directories.
  output_dir = '/Users/KP/Desktop/UfT/CSC2515_Fall_2017/Project/Boston/Boston_Experiments/5-Boston-Calibrated-RTL-No-HP/5-CV-L2-Final-Model'
  quantiles_dir = '/Users/KP/Desktop/UfT/CSC2515_Fall_2017/Project/Boston/Boston_Experiments/5-Boston-Calibrated-RTL-No-HP/5-CV-L2-Final-Quantiles'


  # Create quantiles
  create_quantiles(quantiles_dir, X, y, titles)

  # Create config and then model.
  config = tf.estimator.RunConfig().replace(model_dir=output_dir)

  estimator = create_estimator(config, quantiles_dir)
  
  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  cv_predict = []
  cv_mse = []
  for j, (cv_train_idx, cv_test_idx) in enumerate(kf.split(X)):
    cv_X_train, cv_X_test = X[cv_train_idx], X[cv_test_idx]
    cv_y_train, cv_y_test = y[cv_train_idx], y[cv_test_idx]   
    
    train(estimator, cv_X_train, cv_y_train, titles)
        
    cv_predict.append(predict(estimator, cv_X_test, cv_y_test, titles))
        
    cv_mse.append(mean_squared_error(cv_y_test, predict(estimator, cv_X_test, cv_y_test, titles)))

   
  print("The train CV mean squared errors are:", cv_mse)
  print("The average train CV mean squared error is {}".format(sum(cv_mse)/5))
    
  sns.set(style="whitegrid", palette="pastel", color_codes=True)

  plt.figure(figsize=(10,10))
  plt.title("Boston Dataset House Prices Distribution versus Predictions from 5 Fold RTL Lattice CV with L2")
  plt.xlabel('House Price')
  
  sns.distplot(y, color='g')

  for i,j in zip(range(len(cv_predict)),cv_mse):
    sns.distplot(cv_predict[i], hist=False, kde_kws={"lw": 2, "label": round(j,2)})

   
  plt.show()