import numpy as np
import matplotlib.pyplot as plt
from LinReg import LinReg

np.random.seed(42)
x = 2 * np.random.rand(100,1)
y = 4 + 3*x + np.random.randn(100,1)
model = LinReg()
model.fit(x, y)
y_predict = model.predict(x)
mse = model.mse(y, y_predict)

print(f"Найденный коэффициент w = {model.w:.2f}")
print(f"Найденное смещение b = {model.b:.2f}")
print(mse)

plt.scatter(x, y, color="blue", label = "Данные")
plt.plot(x, y_predict, color = "red", label = "Линия регрессии")
plt.title("Линия регрессии")
plt.xlabel("X (часы учёбы)")
plt.ylabel("y (баллы)")
plt.grid(True)
plt.show()
