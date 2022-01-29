import mglearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR


class PolynomialFeature:
    X, y = mglearn.datasets.make_wave(n_samples=100)
    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    poly = PolynomialFeatures(degree=10, include_bias=False)
    poly.fit(X)
    X_poly = poly.transform(X)

    def main(self):
        print("X_poly.shape: {}".format(self.X_poly.shape))
        print("Entries of X:\n{}".format(self.X[:5]))
        print("Entries of X_poly:\n{}".format(self.X_poly[:5]))
        print("Polynomial feature names:\n{}".format(self.poly.get_feature_names_out()))
        self.linear_regression()
        self.linear_svm()

    def linear_regression(self):
        reg = LinearRegression().fit(self.X_poly, self.y)

        line_poly = self.poly.transform(self.line)
        fig = plt.figure()
        plt.plot(self.line, reg.predict(line_poly), label='polynomial linear regression')
        plt.legend(loc="best")
        plt.ylabel("Regression output")
        plt.xlabel("Input feature")
        plt.plot(self.X[:, 0], self.y, 'o', c='k')
        fig.savefig('polynomial/polynomial_linear_regression.png')

    def linear_svm(self):
        fig = plt.figure()
        for gamma in [1, 10]:
            svr = SVR(gamma=gamma).fit(self.X, self.y)
            plt.plot(self.line, svr.predict(self.line), label='SVR gamma={}'.format(gamma))
        plt.plot(self.X[:, 0], self.y, 'o', c='k')
        plt.ylabel("Regression output")
        plt.xlabel("Input feature")
        plt.legend(loc='best')
        fig.savefig("polynomial/linear_svm.png")
