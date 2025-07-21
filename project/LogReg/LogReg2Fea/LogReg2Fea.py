import numpy as np

class LogReg:
    def __init__(self):
        self.w = None
        self.b = None
        self.lr = 0.1
        self.iter = 10000

    def sigmoid(self,z):
        sigmoid = 1/(1 + np.exp(-z))
        return sigmoid
    
    def predict_proba(self, x):
        y_pred = self.sigmoid(x.dot(self.w)+self.b)
        return y_pred
    
    def predict_class(self,x):
        proba = self.predict_proba(x)
        return (proba>0.5).astype(int)
    
    def fit(self, x, y):
        n_samples,n_features = x.shape
        self.w = np.zeros((n_features,1))
        self.b = 0
        for i in range(self.iter):
            y_pred = self.predict_proba(x)
            dw = (1/n_samples)*x.T.dot(y_pred - y)
            db = (1/n_samples)*np.sum(y_pred - y)
            self.w -= self.lr*dw
            self.b -= self.lr*db
            if i % 1000 == 0:
                loss = self.calc_loss(y, y_pred)
                print(f"Iter {i}: loss = {loss:.4f}, w = {self.w}, b = {self.b}")

    def calc_loss(self, y, y_pred):
        epsilon = 1e-15  # маленькое число, чтобы избежать log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # ограничиваем предсказания
        n = len(y)
        loss = -1/n * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss


    