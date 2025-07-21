from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


data  = load_breast_cancer()
X, y = data .data, data .target

#Кросс-валидация с фиксированным гиперпараметром
knn = KNeighborsClassifier()
scores = cross_val_score(knn, X,y,cv=10)
print("Точность на каждом фолде:", scores)
print("Средняя точность:", np.mean(scores))

#Подбор гиперпараметров с GridSearchCV
param_grid = {'n_neighbors' : [1,2,3,4,5,6,7,8,9,10,11]}
grid_search = GridSearchCV(knn, param_grid, cv=10)
grid_search.fit(X, y)
print("Лучшие параметры:", grid_search.best_params_)
print("Лучшая точность:", grid_search.best_score_)