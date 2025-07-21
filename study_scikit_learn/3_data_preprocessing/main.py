from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

data  = load_breast_cancer()
X, y = data .data, data .target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Модели без масштабирования
knn = KNeighborsClassifier(n_neighbors=5)
logreg = LogisticRegression(max_iter=500)
knn.fit(X_train, y_train)
logreg.fit(X_train, y_train)
print("\nБез масштабирования:")
print(f"KNN: точность = {knn.score(X_test, y_test)}")
print(f"LogReg: точность = {logreg.score(X_test, y_test)}")

# Масштабируем данные
scaler = StandardScaler()
X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
knn.fit(X_train_scaled, y_train)
logreg.fit(X_train_scaled, y_train)
print("\nС масштабированием:")
print(f"KNN: точность = {knn.score(X_test_scaled, y_test)}")
print(f"LogReg: точность = {logreg.score(X_test_scaled, y_test)}")

# Масштабируем данные с помощью pipeline
pipe_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
pipe_logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))

pipe_knn.fit(X_train, y_train)
pipe_logreg.fit(X_train, y_train)
print("\nС масштабированием с помощью pipeline:")
print("KNN:", pipe_knn.score(X_test, y_test))
print("LogReg:", pipe_logreg.score(X_test, y_test))