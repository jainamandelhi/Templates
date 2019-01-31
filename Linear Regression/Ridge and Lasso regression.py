# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 01:06:04 2019

@author: Aman Jain
"""

from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from adspy_shared_utilities import load_crime_dataset
import numpy as np
# Communities and Crime dataset
(X_crime, y_crime) = load_crime_dataset()
#---------------------------------------------------------------------------------------------------

#Ridge Regression
#-----------------------------------------------------------------------------------------------------
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)
linridge = Ridge(alpha=20.0).fit(X_train, y_train)
print('ridge regression linear model intercept: {}'
     .format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'
     .format(linridge.coef_))
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linridge.score(X_test, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))
#---------------------------------------------------------------------------------------------------

#Ridge regression with feature normalization
#---------------------------------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)
print('Crime dataset')
print('ridge regression linear model intercept: {}'
     .format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'
     .format(linridge.coef_))
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linridge.score(X_test_scaled, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))
#-------------------------------------------------------------------------------------------------

#Ridge regression with regularization parameter: alpha
#-------------------------------------------------------------------------------------------------
print('Ridge regression: effect of alpha regularization parameter\n')
    for this_alpha in [0, 1, 10, 20, 50, 100, 1000]:
        linridge = Ridge(alpha = this_alpha).fit(X_train_scaled, y_train)
        r2_train = linridge.score(X_train_scaled, y_train)
        r2_test = linridge.score(X_test_scaled, y_test)
        num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
        print('Alpha = {:.2f}\nnum abs(coeff) > 1.0: {}, \
        r-squared training: {:.2f}, r-squared test: {:.2f}\n'
        .format(this_alpha, num_coeff_bigger, r2_train, r2_test))
#---------------------------------------------------------------------------------------------
        
#Lasso Regularization
#----------------------------------------------------------------------------------------------
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linlasso = Lasso(alpha=2.0, max_iter = 10000).fit(X_train_scaled, y_train)

print('Crime dataset')
print('lasso regression linear model intercept: {}'
     .format(linlasso.intercept_))
print('lasso regression linear model coeff:\n{}'
     .format(linlasso.coef_))
print('Non-zero features: {}'
     .format(np.sum(linlasso.coef_ != 0)))
print('R-squared score (training): {:.3f}'
     .format(linlasso.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}\n'
     .format(linlasso.score(X_test_scaled, y_test)))
print('Features with non-zero weight (sorted by absolute magnitude):')
#-----------------------------------------------------------------------------------------------

#Lasso regression: effect of alpha regularization
#----------------------------------------------------------------------------------------------
for alpha in [0.5, 1, 2, 3, 5, 10, 20, 50]:
    linlasso = Lasso(alpha, max_iter = 10000).fit(X_train_scaled, y_train)
    r2_train = linlasso.score(X_train_scaled, y_train)
    r2_test = linlasso.score(X_test_scaled, y_test)
    
    print('Alpha = {:.2f}\nFeatures kept: {}, r-squared training: {:.2f}, \
          r-squared test: {:.2f}\n'
         .format(alpha, np.sum(linlasso.coef_ != 0), r2_train, r2_test))
#--------------------------------------------------------------------------------------------