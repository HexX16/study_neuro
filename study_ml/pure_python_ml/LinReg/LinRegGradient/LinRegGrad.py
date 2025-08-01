import numpy as np

class LinRegGrad():
    def __init__(self):
        self.w = None
        self.b = None
        self.lr = 0.00001
        self.epochs = 10000
    def predict(self, x):
        return self.w*x+self.b
    def fit(self, x, y):
        self.w = 0
        self.b = 0
        n = len(y)
        for _ in range(self.epochs):
            y_pred = self.predict(x)
            dw = (-2/n)*np.sum(x * (y - y_pred))
            db = (-2/n)*np.sum(y - y_pred)
            self.w -= dw*self.lr
            self.b -= db*self.lr
    def rmse(self, y, y_pred):
        n=len(y)
        rmse = np.sqrt((1/n) * np.sum((y - y_pred) ** 2))
        print(f"\nОшибка(тыс.руб):{int(rmse)}")
