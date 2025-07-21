import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from termcolor import colored

cv = KFold(n_splits=2, shuffle=True, random_state=42)

data = pd.read_csv('price_moscow.csv')
cost = data['Price']
data = data.drop(columns=['Price'])
cost_log = np.log1p(cost)

num_cols = ['Minutes to metro', 'Number of rooms', 'Area', 'Living area', 'Kitchen area', 'Floor', 'Number of floors', 'AVERAGE']
cat_cols = ['Apartment type', 'Metro station', 'Region', 'Renovation']

num_pipeline = Pipeline([   
    ('scaler', MinMaxScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

data_processor = preprocessor.fit_transform(data)
if hasattr(data_processor, "toarray"):
    data_processor = data_processor.toarray()
cat_cols_encoded = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
all_cols = num_cols + list(cat_cols_encoded)
data = pd.DataFrame(data_processor, columns=all_cols, index=data.index)

X_train, X_test, y_train, y_test = train_test_split(data, cost_log, test_size=0.2, random_state=42)

models = {
    'Ridge': (Ridge(), {
        'alpha': [0.01, 0.1, 1, 10],
        'solver': ['auto', 'saga', 'svd']
    }),
    'Lasso': (Lasso(max_iter=10000), {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1]
    }),
    'ElasticNet': (ElasticNet(max_iter=10000), {
        'alpha': [0.001, 0.01, 0.1, 1],
        'l1_ratio': [0.2, 0.5, 0.8]
    }),
    'RandomForest': (RandomForestRegressor(random_state=42), {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None]
    }),
    'GradientBoosting': (GradientBoostingRegressor(random_state=42), {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }),
}

best_models = {}

for name, (model, param) in models.items():
    print(colored(f"\n\nМодель: {name}", "red"))
    print("Без подбора гиперпараметров:")
    scores_model = cross_val_score(model, data, cost_log, cv=cv, scoring='neg_mean_squared_error')
    neg_mse_model = np.mean(scores_model)
    rmse_model = np.sqrt(-neg_mse_model)
    exp_rmse_model = np.exp(rmse_model)
    print(f"neg_mse = {neg_mse_model:.2f}")
    print(f"rmse = {rmse_model:.2f}")
    print(f"Во сколько раз модель ошибается в среднем (exp_rmse) = {exp_rmse_model:.2f} = {(exp_rmse_model-1)*100:.2f}%")
    grid = GridSearchCV(model, param, cv = cv, scoring='neg_mean_squared_error')
    grid.fit(data, cost_log)
    best_model = grid.best_estimator_
    best_models[name] = best_model
    print("С подбором гиперпараметров:")
    scores_best_model = cross_val_score(best_model, data, cost_log, cv=cv, scoring='neg_mean_squared_error')
    neg_mse_best_model = np.mean(scores_best_model)
    rmse_best_model = np.sqrt(-neg_mse_best_model)
    exp_rmse_best_model = np.exp(rmse_best_model)
    print(f"neg_mse = {neg_mse_best_model:.2f}")
    print(f"rmse = {rmse_best_model:.2f}")
    print(f"Во сколько раз модель ошибается в среднем (exp_rmse) = {exp_rmse_best_model:.2f} = {(exp_rmse_best_model-1)*100:.2f}%")


for name, model in best_models.items():
    print(colored(f"\n\nМодель: {name}", "red"))
    model.fit(X_train, y_train)
    predicts = model.predict(X_test)
    mse = mean_squared_error(y_test, predicts)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predicts)
    print(f"MSE = {mse:.2f}")
    print(f"RMSE = {rmse:.2f}")
    print(f"r2 = {r2:.2f}")
    