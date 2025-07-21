import numpy as np
import matplotlib.pyplot as plt
from LogReg4Fea import LogReg

# Признаки: [мощность, объем (л), вес (кг), аэродинамика (Cd)]
x = np.array([
    [450, 5.0, 1720, 0.36],  # Ford Mustang GT
    [255, 2.0, 1550, 0.27],  # BMW 330i
    [310, 2.5, 1550, 0.28],  # Subaru WRX STI
    [180, 1.5, 1370, 0.30],  # Honda Civic Turbo
    [354, 3.0, 1690, 0.31],  # Audi S4 (замена Tesla Model 3 LR)
    [228, 2.0, 1400, 0.29],  # VW Golf GTI
    [300, 3.6, 1700, 0.31],  # Dodge Charger
    [192, 2.5, 1450, 0.29],  # Mazda 6
    [184, 2.0, 1420, 0.30],  # Hyundai Elantra N
    [280, 3.3, 1650, 0.27],  # Kia Stinger
])

# Цель: 1 — ≤ 6 сек до 100 км/ч, 0 — медленнее
y = np.array([1, 1, 1, 0, 1, 1, 1, 0, 0, 1])

# Названия моделей (для визуализации)
models = [
    "Ford Mustang GT",
    "BMW 330i",
    "Subaru WRX STI",
    "Honda Civic Turbo",
    "Audi S4",
    "VW Golf GTI",
    "Dodge Charger",
    "Mazda 6",
    "Hyundai Elantra N",
    "Kia Stinger"
]

x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
x_norm = (x-x_mean)/x_std

model = LogReg()
model.fit(x_norm, y)
y_pred = model.predict_class(x_norm)

plt.scatter(models, y, color = "blue")
plt.scatter(models, y_pred, color = "red")
plt.xlabel("Модели авто")
plt.ylabel("1 — ≤ 6 сек до 100 км/ч, 0 — медленнее")
plt.grid(True)
plt.show()