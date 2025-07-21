import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from LinRegGrad import LinRegGrad

np.random.seed(0)
x = np.random.randint(20,300, (100,1))
y = 10000*x + np.random.randint(100,200000,(100,1))

model = LinRegGrad()
model.fit(x,y)
# Текущие предсказания
y_pred = model.predict(x)
model.rmse(y,y_pred)

# Создаём новые значения x для прогноза
x_new = np.linspace(x.min(), x.max() + 50, 200).reshape(-1, 1)  # Например, до max+50
# Предсказания на новых x
y_new_pred = model.predict(x_new)

model_sklearn = LinearRegression()
model_sklearn.fit(x,y)
y_pred_sklearn = model_sklearn.predict(x)

print("My Linear Regression:")
print(f"  w = {model.w:.4f}, b = {model.b:.4f}")
print(f"  MSE = {mean_squared_error(y, y_pred):.4f}")

print("\nsklearn LinearRegression:")
print(f"  w = {model_sklearn.coef_[0][0]:.4f}, b = {model_sklearn.intercept_[0]:.4f}")
print(f"  MSE = {mean_squared_error(y, y_pred_sklearn):.4f}")

plt.plot(x_new, y_new_pred, color='red')
plt.scatter(x, y, color='blue', marker='o')
plt.xlabel('x (площадь)')
plt.ylabel('y (цена)')
plt.title('Прогноз цены от площади')
plt.grid(True)
plt.show()