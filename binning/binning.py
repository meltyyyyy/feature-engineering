import mglearn.datasets
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


class Binning:
    X, y = mglearn.datasets.make_wave(n_samples=100)
    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

    def main(self):
        self.showFigure()
        self.execute()

    def showFigure(self):
        fig = plt.figure()

        reg = DecisionTreeRegressor(min_samples_split=3).fit(self.X, self.y)
        plt.plot(self.line, reg.predict(self.line), label="Decision Tree")

        reg = LinearRegression().fit(self.X, self.y)
        plt.plot(self.line, reg.predict(self.line), label="Linear Regression")

        plt.plot(self.X[:, 0], self.y, 'o', c='k')
        plt.ylabel("Regression output")
        plt.xlabel("Input feature")
        plt.legend(loc="best")
        fig.savefig('binning/regressions.png')

    def execute(self):
        bins = np.linspace(-3, 3, 11)
        print("bins: {}".format(bins))

        which_bin = np.digitize(self.X, bins=bins)
        print("\nData points:\n", self.X[:5])
        print("\nBin membership for data points:\n", which_bin[:5])

        encoder = OneHotEncoder(sparse=False)
        encoder.fit(which_bin)
        X_binned = encoder.transform(which_bin)
        print(X_binned[:5])

        print("X_binned.shape: {}".format(X_binned.shape))


