# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 00:10:41 2019

@author: Aman Jain
"""
"""
DATA PROCESSING
"""
#.............................................................................................
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

fruits = pd.read_table('fruit_data_with_colors.txt')

fruits.head()

list(fruits)

fruitnames=dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique()))

X = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8, random_state=0)
#........................................................................................

"""
SCATTERPLOT MATRIX
"""
#.................................................................................................
from matplotlib import cm
cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)
#...................................................................................................

"""
KNN CLASSIFIER
"""
#............................................................................................
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 4,weights='uniform')

knn.fit(X_train, y_train)

knn.score(X_test,y_test)

fruit_prediction = knn.predict([[20, 4.3, 5.5, 0.52]])
fruit_prediction
fruitnames[fruit_prediction[0]]
#.........................................................................................

"""
DEPICTING GRAPH OF BOUNDARIES FOR CLASSIFYING DIFFERENT DATA POINTS
"""
#...........................................................................................
from adspy_shared_utilities import plot_fruit_knn

plot_fruit_knn(X_train, y_train, 5, 'uniform')   # we choose 5 nearest neighbors
#.........................................................................................

"""
GRAPH DEPICTING EFFECT OF ACCURACY ON NEAREST NEIGHBOURS
"""
#.........................................................................................
k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k,weights='uniform')
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);
#.........................................................................................

"""
GRAPH DEPICTING EFFECT ON TRAIN_TEST_SPLIT RATIO AT A FIXED K
"""
#.......................................................................................
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

knn = KNeighborsClassifier(n_neighbors = 4,weights='uniform')

plt.figure()

for s in t:

    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy');
#.......................................................................................

"""
COMPARING TRAIN AND TEST SCORE TO DECREASE OVERFITTING
"""
#......................................................................................
from adspy_shared_utilities import plot_two_class_knn

plot_two_class_knn(X_train.values[:,0:2], y_train.values, 1, 'uniform', X_test.values[:,0:2], y_test.values)
plot_two_class_knn(X_train.values[:,0:2], y_train.values, 3, 'uniform', X_test.values[:,0:2], y_test.values)
plot_two_class_knn(X_train.values[:,0:2], y_train.values, 7, 'uniform', X_test.values[:,0:2], y_test.values)
#.............................................................................................................