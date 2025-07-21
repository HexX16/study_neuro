import numpy as np

class LogReg:
    def __init__(self):
        self.w = None
        self.b = None
        self.lr = 0.0001
        self.epochs = 2000
    
    def sigmoid(self, z):
        sigmoid = 1/(1 + np.exp(-z))
        return sigmoid

    def predict_proba(self, x):
        y_pred = self.sigmoid(x.dot(self.w) + self.b)
        return y_pred
    
    def predict_class(self,x):
        proba = self.predict_proba(x)
        return (proba>0.5).astype(int)
    
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.b = 0
        self.w = np.zeros((n_features, 1))
        for iter in range(self.epochs):
            y_pred = self.predict_proba(x)
            dw = (1/n_samples)*x.T.dot(y_pred - y.reshape(-1,1))
            db = (1/n_samples)*np.sum(y_pred - y.reshape(-1,1))
            self.w -= dw * self.lr
            self.b -= db * self.lr
            if iter%100==0:
                loss = self.calc_loss(y, y_pred)
                print(f"w={self.w}, b={self.b}, mean_loss={loss}")
    
    def calc_loss(self, y, y_pred):
        eps = 1e-15  # защита от log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
        mean_loss = np.mean(loss)
        return mean_loss
