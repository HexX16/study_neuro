import numpy as np

class LogReg:
    def __init__(self):
        self.w = None
        self.b = None
        self.lr = 0.1
        self.n_iter = 10000

    def sigmoid(self,z):
        sigmoid = 1/(1 + np.exp(-z))
        return sigmoid
    
    def predict_proba(self, x):
        z = x.dot(self.w)+self.b
        return self.sigmoid(z)

    def predict_class(self,x):
        proba = self.predict_proba(x)
        return (proba>0.5).astype(int)
        
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.w = np.zeros((n_features,1))
        self.b = 0
        for i in range(self.n_iter):
            z = x.dot(self.w) + self.b
            y_predict = self.sigmoid(z).reshape(-1, 1)  # (100,1)
            dw = (1 / n_samples) * x.T.dot(y_predict - y)  # (1,100) * (100,1) -> (1,1)
            db = (1 / n_samples) * np.sum(y_predict - y)
            self.w -= self.lr * dw.flatten()
            self.b -= self.lr * db
            if i % 1000 == 0:
                y_pred = self.predict_proba(x)
                loss = self.calc_loss(y, y_pred)
                print(f"Iter {i}: loss = {loss:.4f}, w = {self.w}, b = {self.b}")

    def calc_loss(self, y, y_pred):
        epsilon = 1e-15  # маленькое число, чтобы избежать log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # ограничиваем предсказания
        n = len(y)
        loss = -1/n * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss


