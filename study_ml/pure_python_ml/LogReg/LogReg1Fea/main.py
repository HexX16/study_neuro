import numpy as np
import matplotlib.pyplot as plt
from LogReg import LogReg

# Данные
np.random.seed(0)
x = np.linspace(0, 10, 100).reshape(-1, 1)
y = (x > 5).astype(int).reshape(-1, 1)

# Обучение
model = LogReg()
model.fit(x, y)

# Предсказания
y_pred = model.predict_class(x)

# Визуализация

plt.scatter(x, y, label="Истинные метки", alpha=0.4)
plt.plot(x, model.predict_proba(x), color='red', label="Предсказанная вероятность")
plt.xlabel("x")
plt.ylabel("y (вероятность)")
plt.legend()
plt.title("Логистическая регрессия")
plt.grid(True)
plt.show()
