import mglearn.datasets
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


class Binning:
    X, y = mglearn.datasets.make_wave(n_samples=100)
    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    bins = np.linspace(-3, 3, 11)

    def main(self):
        self.showFigure()
        self.execute()
        self.regression_with_binned_data()

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
        print("bins: {}".format(self.bins))

        which_bin = np.digitize(self.X, bins=self.bins)
        print("\nData points:\n", self.X[:5])
        print("\nBin membership for data points:\n", which_bin[:5])

        encoder = OneHotEncoder(sparse=False)
        encoder.fit(which_bin)
        X_binned = encoder.transform(which_bin)
        print(X_binned[:5])

        print("X_binned.shape: {}".format(X_binned.shape))

    def regression_with_binned_data(self):
        which_bin = np.digitize(self.X, bins=self.bins)
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(which_bin)
        X_binned = encoder.transform(which_bin)
        line_binned = encoder.transform(np.digitize(self.line,bins=self.bins))

        fig = plt.figure()

        reg = LinearRegression().fit(X_binned,self.y)
        plt.plot(self.line,reg.predict(line_binned),label="linear regression binned")

        reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned,self.y)
        plt.plot(self.line,reg.predict(line_binned),label="decision tree binned")
        plt.plot(self.X[:,0],self.y,'o',c='k')
        plt.vlines(self.bins,-3,3,linewidth = 1,alpha=.2)
        plt.legend(loc="best")
        plt.ylabel("Regression input")
        plt.xlabel("Input feature")
        fig.savefig('binning/binned_data_regression.png')
