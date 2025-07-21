import numpy as np
import matplotlib.pyplot as plt

class LinReg:
    def __init__(self):
        self.w = None
        self.b = None
    def fit (self, x, y):
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov = np.sum((x-x_mean)*(y-y_mean)) #ковариация. Ковариация — это мера, которая показывает, насколько два признака (например, x и y) меняются одновременно.
        var = np.sum((x-x_mean)**2) #дисперсия. Дисперсия показывает, насколько сильно данные разбросаны относительно своего среднего.
        self.w = cov/var
        self.b = y_mean - self.w * x_mean

    #Предсказываем y
    def predict(self,x):
        return self.w * x + self.b
    
    #MSE(Mean Squared Error) — это среднеквадратичная ошибка, один из самых популярных способов оценить качество модели.
    def mse (self, y, y_predict):
        return np.mean((y - y_predict)**2)

