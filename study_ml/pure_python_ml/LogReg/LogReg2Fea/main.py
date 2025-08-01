import numpy as np
import matplotlib.pyplot as plt
from LogReg2Fea import LogReg

np.random.seed(0)
x1 = np.linspace(0, 10, 50)
x2 = np.linspace(10, 20, 50)
x = np.column_stack((x1, x2))
y = (x1 + x2 > 25).astype(int).reshape(-1, 1)

model = LogReg()
model.fit(x, y)

# Отрисуем исходные точки с цветом по классам
plt.scatter(x1[y.flatten() == 0], x2[y.flatten() == 0], color='blue', label='Класс 0')
plt.scatter(x1[y.flatten() == 1], x2[y.flatten() == 1], color='red', label='Класс 1')

# Создадим сетку значений для обоих признаков
x1_grid, x2_grid = np.meshgrid(np.linspace(0, 10, 100), np.linspace(10, 20, 100))
grid_points = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))

# Предсказанные вероятности на сетке
probs = model.predict_proba(grid_points).reshape(x1_grid.shape)

# Нарисуем контур вероятностей
contour = plt.contour(x1_grid, x2_grid, probs, levels=[0.1, 0.3, 0.5, 0.7, 0.9], cmap='Greys')
plt.clabel(contour, inline=True, fontsize=8)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Вероятность класса 1, предсказанная логистической регрессией')
plt.legend()
plt.grid(True)
plt.show()
