from sklearn.datasets import load_wine
import pandas as pd
import numpy as np 
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from termcolor import colored
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
data = load_wine()
cv = KFold(n_splits=5, shuffle=True, random_state=42)

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
type = df['target']
df = df.drop(columns=['target'], axis = 0)
X_train, X_test, y_train, y_test = train_test_split(df, type, test_size=0.2, random_state=42)

num_cols = list(df.columns)

num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols)
])

df_processor = preprocessor.fit_transform(df)
df_processor = pd.DataFrame(df_processor, columns=df.columns)

best_models = {}
models = {
    'Logistic Regression': (
        LogisticRegression(max_iter=10000),
        {
            'C': [0.01, 0.1, 1, 10, 100],            # коэффициент регуляризации (чем меньше - сильнее регуляризация)
            'penalty': ['l2'],                  # тип регуляризации: L1 — Lasso, L2 — Ridge
        }
    ),
    'Random Forest': (
        RandomForestClassifier(),
        {
            'n_estimators': [50, 100, 200],           # количество деревьев в лесу
            'max_depth': [None, 10, 20, 30],          # максимальная глубина дерева (None — без ограничения)
            'min_samples_split': [2, 5, 10],          # минимальное число образцов для разбиения узла
            'min_samples_leaf': [1, 2, 4]              # минимальное число образцов в листовом узле
        }
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(),
        {
            'n_estimators': [100, 200],                # количество слабых моделей (деревьев)
            'learning_rate': [0.01, 0.1, 0.2],         # скорость обучения (чем меньше — тем медленнее обучение)
            'max_depth': [3, 5, 7],                    # максимальная глубина каждого дерева
            'subsample': [0.8, 1.0]                    # доля выборки для построения каждого дерева (для стохастичности)
        }
    ),
    'Decision Tree': (
        DecisionTreeClassifier(),
        {
            'max_depth': [None, 10, 20, 30],           # максимальная глубина дерева
            'min_samples_split': [2, 5, 10],           # минимальное число образцов для разбиения
            'min_samples_leaf': [1, 2, 4]               # минимальное число образцов в листовом узле
        }
    ),
    'SVM': (
        SVC(probability=True),
        {
            'C': [0.1, 1, 10, 100],                     # коэффициент регуляризации (больше — меньше регуляризации)
            'kernel': ['linear', 'rbf'],                # ядро: линейное или радиальное базисное (RBF)
            'gamma': ['scale', 'auto']                   # параметр ядра для rbf ('scale' — 1/(n_features*X.var))
        }
    ),
    'K-Nearest Neighbors': (
        KNeighborsClassifier(),
        {
            'n_neighbors': [3, 5, 7, 9],                 # количество соседей для классификации
            'weights': ['uniform', 'distance'],          # вес соседей: одинаковый или обратно пропорционален расстоянию
            'metric': ['euclidean', 'manhattan']         # метрика расстояния: Евклидова или Манхэттенская
        }
    )
}

for name, (model, param) in models.items():
    print(colored(f"\n{name}:", 'red'))
    accuracy_model = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1, scoring='accuracy')
    mean_accuracy_model = np.mean(accuracy_model)
    print(colored("Без подбора гиперпараметров:", 'yellow'))
    print(f"Средняя точность (train, CV) = {mean_accuracy_model:.3f}")
    grid = GridSearchCV(model, param, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model
    accuracy_best_model = cross_val_score(best_model, X_train, y_train, cv=cv, n_jobs=-1, scoring='accuracy')
    mean_accuracy_best_model = np.mean(accuracy_best_model)
    print(colored("С подбором гиперпараметров:", 'yellow'))
    print(f"Средняя точность (train, CV) = {mean_accuracy_best_model:.3f}")

print("\n\n\n\nФинальная оценка на test:")
for name, model in best_models.items():
    print(colored(f"\n{name}:", 'red'))
    model.fit(X_train, y_train)
    predicts = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicts)
    print(f"Точность = {accuracy:.3f}")

    # Матрица ошибок
    cm = confusion_matrix(y_test, predicts)
    print(f"Матрица ошибок:\n{cm}")

    print(classification_report(y_test, predicts))
    
    





