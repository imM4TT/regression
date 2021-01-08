import numpy as np


class LogisticRegression:

    def __init__(self, x, y):
        self.m = len(y)
        self.X = x
        self.Y = y.reshape((self.m, 1))
        self.theta = [np.random.randn(self.X.shape[1], 1), np.random.randn(1)]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        pred = self.get_predictions(x)
        return (pred >= 0.5).astype(int)

    def get_predictions(self, x):
        Z = x.dot(self.theta[0]) + self.theta[1]
        return self.sigmoid(Z)

    # Fonction de cout qui évalue notre modèle: l'erreur q. moyenne
    def cost_function(self, pred_y):
        return (1 / self.m) * np.sum(-self.Y * np.log(pred_y) - (1 - self.Y) * np.log(1 - pred_y))

    # Calcul de dérivée de la fonction de cout pour connaitre l'inclinaison de la pente
    def get_gradient(self, pred_y):
        dW = 1 / self.m * np.dot(self.X.T, pred_y - self.Y)
        db = 1 / self.m * np.sum(pred_y - self.Y)
        return dW, db

    def gradient_descent(self, n, learning_rate):
        pred_y = self.get_predictions(self.X)
        cost_history = self.cost_function(pred_y)
        for i in range(n - 1):
            dW, db = self.get_gradient(pred_y)
            self.theta[0] = self.theta[0] - (learning_rate * dW)
            self.theta[1] = self.theta[1] - (learning_rate * db)
            pred_y = self.get_predictions(self.X)
            cost_history = np.vstack((cost_history, self.cost_function(pred_y)))
        return cost_history
