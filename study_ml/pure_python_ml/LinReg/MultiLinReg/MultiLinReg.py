import numpy as np

class MultiLinReg:
    def __init__(self):
        self.w = None
        self.b = None
        self.lr = 0.01
        self.epochs = 10000
    
    def predict(self, x):
        return x.dot(self.w)+self.b
    
    def fit(self, x, y):
        n, m = x.shape
        self.w = np.zeros(m)  # инициализация весов нулями
        self.b = 0
        for i in range(self.epochs):
            y_pred = self.predict(x)
            dw = (-2/n)*x.T.dot(y - y_pred)
            db = (-2/n)*np.sum(y - y_pred)
            self.w -= dw*self.lr
            self.b -= db*self.lr
            if i%1000 == 0:
                print(f"Вес: {self.w}")