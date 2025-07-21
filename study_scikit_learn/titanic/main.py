from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


titanic = fetch_openml('titanic', version=1, as_frame=True)
X = titanic.data
y = titanic.target
X = X.drop(columns = ['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest'])

num_cols = ['age', 'sibsp', 'parch', 'fare']
cat_cols = ['pclass', 'sex', 'embarked']

# Пайплайн для числовых признаков
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

# Пайплайн для категориальных признаков
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Объединяю пайплайны
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline,cat_cols)
])

X_processed = preprocessor.fit_transform(X)

# Получаем имена колонок после one-hot кодирования
cat_cols_encoded = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)

# Формируем DataFrame из результата
all_cols = num_cols + list(cat_cols_encoded)
X = pd.DataFrame(X_processed, columns=all_cols, index=X.index)

# Создаем начальную модель(до подбора гиперпараметров):
model = LogisticRegression()
# Точность по кросс-валидации(до подбора гиперпараметров):
scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print(scores)
print(f"Среднее(до подбора гиперпараметров): {np.mean(scores):.2f}\n")

# Подбор гиперпараметров:
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}
grid = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', n_jobs=1)
grid.fit(X, y)
print("Лучшие параметры:", grid.best_params_)
print("Лучшая точность:", grid.best_score_, "\n")

# Создаем финальную модель(после подбора гиперпараметров):
final_model = LogisticRegression(C=0.01, solver='lbfgs')
# Точность по кросс-валидации(после подбора гиперпараметров):
scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print(f"Среднее(после подбора гиперпараметров): {np.mean(scores):.2f}\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
accuracy = final_model.score(X_test, y_test)
print(f"Финальная точность(с примененными гиперпараметрами): {accuracy:.2f}\n")

# Оценю важность каждого признака:
coeff = final_model.coef_[0]
features = X.columns
importance_df = pd.DataFrame({
    'Признак': features,
    'Вес (коэф.)': coeff
})
print(importance_df)
