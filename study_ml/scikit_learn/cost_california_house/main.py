from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


houses = fetch_california_housing(as_frame=True).frame
cost = houses['MedHouseVal']
cost*=10000
houses = houses.drop(columns = ['MedHouseVal'])
houses['MedInc']*=10000

scaler = MinMaxScaler()
houses = pd.DataFrame(scaler.fit_transform(houses), columns = houses.columns)
cv = KFold(n_splits=10, shuffle=True, random_state=42)
# Создаём модель без регуляризации:
linreg = LinearRegression()
scores = cross_val_score(linreg, houses, cost, cv = cv, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"RMSE по фолдам: {rmse_scores}")
print(f"Средняя RMSE без регуляризации: {np.mean(rmse_scores):.2f}")

X_train, X_test, y_train, y_test = train_test_split(houses, cost, test_size=0.2, random_state=42)
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)
accuracy = linreg.score(X_test, y_test)
print(f"Точноть без регуляризации: {accuracy:.4f}")

# Создаём модель с регуляризацией
linreg_ridge = Ridge(alpha=1.0)
scores = cross_val_score(linreg_ridge, houses, cost, cv = cv, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"RMSE по фолдам: {rmse_scores}")
print(f"Средняя RMSE с регуляризацией (без подбора гиперпараметров): {np.mean(rmse_scores):.2f}")

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['auto', 'svd', 'cholesky', 'saga']
}
grid = GridSearchCV(linreg_ridge, param_grid, cv = cv, scoring='neg_mean_squared_error')
grid.fit(houses,cost)
print(f"Лучшие параметры: {grid.best_params_}")
best_rmse = np.sqrt(-grid.best_score_)
print(f"Лучший RMSE с регуляризацией (этап подбора гиперпараметров): {best_rmse:.2f}")

linreg_ridge_final = Ridge(alpha=0.1, solver='saga')
linreg_ridge_final.fit(X_train,y_train)
y_pred = linreg_ridge_final.predict(X_test)
score = linreg_ridge_final.score(X_test, y_test)
print(f"Точноть c регуляризацией: {score:.4f}")