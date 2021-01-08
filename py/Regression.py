import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score


class Regression:

    def __init__(self, x, y):
        self.m = len(y)
        self.X = x
        self.X = np.hstack((np.ones((self.m, 1)), self.X))  # ajout d'une colonne biais
        self.Y = y.reshape((self.m, 1))
        self.theta = np.random.randn(self.X.shape[1], 1)
        self.y_model = None

    # Produit matriciel qui met a jour toutes les valeurs y de notre fonction
    def update_model(self):
        self.y_model = self.X.dot(self.theta)

    # Fonction de cout qui évalue notre modèle: le nombre d'erreur q. moyen
    def cost_function(self):
        return 1 / (2 * self.m) * np.sum((self.y_model - self.Y) ** 2)

    # Calcul de dérivée de la fonction de cout pour connaitre l'inclinaison de la pente
    def get_gradient(self):
        return 1 / self.m * self.X.T.dot(self.y_model - self.Y)

    def gradient_descent(self, n, learning_rate):
        self.update_model()
        cost_history = self.cost_function()
        model_y_history = self.y_model

        for i in range(n - 1):
            self.theta = self.theta - (learning_rate * self.get_gradient())
            self.update_model()
            cost_history = np.vstack((cost_history, self.cost_function()))
            model_y_history = np.hstack((model_y_history, self.y_model))

        return self.theta, self.y_model, model_y_history, cost_history


class Utils:

    # Création d'un dataset linéaire
    @staticmethod
    def get_data_set():
        n_samples = 5  # nombre d'echantillons a générer
        x = np.linspace(0, 10, n_samples)
        y = - (x + np.random.randn(n_samples))
        return x, y

    # Affichage, debug
    @staticmethod
    def plot_result(x, y, pred, theta, n_iter, model_hist, cost_hist):
        # Évolution de la Fonction de cout
        plt.suptitle('Diminution de la fonction de cout', fontsize=20)
        plt.xlabel("n", fontsize=14)
        plt.ylabel("Taux d'erreur quadratique moyen: \nfonction de cout", fontsize=14)
        plt.plot(range(n_iter), cost_hist)
        plt.show()

        # Évolution de la régression
        plt.scatter(x, y)
        plt.plot(x, model_hist[:, ::100], linewidth=1)
        plt.suptitle('Évolution de la régression', fontsize=20)
        plt.xlabel('x', fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.show()

        # Régression finale
        r2 = str(round(r2_score(y, pred), 2))
        red_patch = mpatches.Patch(color='red')
        plt.legend(handles=[red_patch])
        whitePatch = mpatches.Rectangle((0, 0), 0, 0, color='white')
        plt.scatter(x, y)
        plt.legend([whitePatch, whitePatch], ['y=' + str(theta[-1][0]) + "x + " + str(sum(theta[:-1][0])), "R2 = " + r2],
                   markerscale=1, loc="upper right", frameon=True, fontsize=10)

        plt.scatter(x, pred, c='r', s=30)
        plt.show()

    # region Driver code Test
    @staticmethod
    def start_test():
        # np.random.seed(0)  # pour toujours reproduire le meme dataset
        n_iter = 1000
        ln_rate = .01
        # x, y = Utils.get_data_set()
        df = pd.read_csv("data/carData.csv")
        df["Year"] = 2020 - df["Year"]
        df["Kms_Driven"] = 1/df["Kms_Driven"]

        x = np.array(df[["Year", "Kms_Driven"]])
        y = np.array(df[["Selling_Price"]])

        reg = Regression(x, y)
        theta, pred_y, model_hist, cost_hist = reg.gradient_descent(n_iter, ln_rate)

        Utils.plot_result(x[:, 0], y, pred_y, theta, n_iter, model_hist, cost_hist)

    # endregion


# Utils.start_test()
