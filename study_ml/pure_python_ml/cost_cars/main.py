import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# Загрузка CSV
df = pd.read_csv("cars.csv")

df['car_age'] = 2025 - df['year']
df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)
df = df.drop('year', axis=1)
df = df.drop('name', axis=1)
# Фильтрация по цене
lower_limit = df['selling_price'].quantile(0.01)
upper_limit = df['selling_price'].quantile(0.99)
df = df[(df['selling_price'] > lower_limit) & (df['selling_price'] < upper_limit)]


X = df.drop('selling_price', axis = 1)
y = df['selling_price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

