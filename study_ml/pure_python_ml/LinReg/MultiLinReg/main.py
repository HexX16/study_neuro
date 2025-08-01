import numpy as np
import matplotlib.pyplot as plt
from MultiLinReg import MultiLinReg

import numpy as np

# Обновлённые признаки (x): [л.с., объем двигателя (л), вес (кг)]
x = np.array([
    [203, 2.5, 1570],   # Toyota Camry 2.5
    [255, 2.0, 1550],   # BMW 330i
    [201, 2.0, 1500],   # Audi A4 2.0
    [450, 5.0, 1720],   # Ford Mustang GT
    [180, 1.5, 1370],   # Honda Civic 1.5
    [255, 2.0, 1600],   # Mercedes C300
    [228, 2.0, 1400],   # VW Golf GTI
    [310, 2.5, 1550],   # Subaru WRX STI
    [188, 2.5, 1500],   # Nissan Altima 2.5
    [300, 3.6, 1700],   # Dodge Charger
    [306, 3.5, 1690],   # Toyota Avalon
    [280, 3.3, 1650],   # Kia Stinger GT-Line
    [184, 2.0, 1420],   # Hyundai Elantra N
    [192, 2.5, 1450],   # Mazda 6 2.5
])

# Время разгона до 100 км/ч (сек)
y = np.array([
    7.6,  # Toyota Camry
    5.6,  # BMW 330i
    7.3,  # Audi A4
    4.3,  # Mustang GT
    7.8,  # Honda Civic
    5.9,  # Mercedes C300
    6.4,  # Golf GTI
    5.2,  # WRX STI
    7.9,  # Altima
    5.5,  # Dodge Charger
    6.1,  # Toyota Avalon
    6.3,  # Kia Stinger
    7.4,  # Elantra N
    7.1,  # Mazda 6
])

# Названия моделей
models = [
    "Toyota Camry 2.5",
    "BMW 330i",
    "Audi A4 2.0 TFSI",
    "Ford Mustang GT",
    "Honda Civic 1.5 Turbo",
    "Mercedes C300",
    "Volkswagen Golf GTI",
    "Subaru WRX STI",
    "Nissan Altima 2.5",
    "Dodge Charger 3.6",
    "Toyota Avalon 3.5",
    "Kia Stinger GT-Line",
    "Hyundai Elantra N",
    "Mazda 6 2.5"
]

# Считаем среднее и стандартное отклонение по каждому признаку (по колонкам)
mean = np.mean(x, axis=0)
std = np.std(x, axis=0)
# Нормализация
x_norm = (x - mean) / std


model = MultiLinReg()
model.fit(x_norm, y)
y_pred = model.predict(x_norm)

plt.scatter(models, y, color="blue", label = "Реальные значения")
plt.scatter(models, y_pred, color = "red", label = "Предсказанные значения")
plt.xlabel("Модели")
plt.ylabel("Разгон до 100км/ч")
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()