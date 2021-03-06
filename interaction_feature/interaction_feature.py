import matplotlib.pyplot as plt
import mglearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


class InteractionFeature:
    X, y = mglearn.datasets.make_wave(n_samples=100)
    bins = np.linspace(-3, 3, 11)
    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    which_bin = np.digitize(X, bins=bins)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(which_bin)
    X_binned = encoder.transform(which_bin)
    line_binned = encoder.transform(np.digitize(line, bins=bins))
    X_combined = np.hstack([X, X_binned])
    X_product = np.hstack([X_binned,X*X_binned])

    def main(self):
        self.bin_combined()
        self.bin_product()

    def bin_combined(self):
        reg = LinearRegression().fit(self.X_combined, self.y)

        line_combined = np.hstack([self.line, self.line_binned])
        fig = plt.figure()
        plt.plot(self.line, reg.predict(line_combined), label='linear regression combined')

        for bin in self.bins:
            plt.plot([bin, bin], [-3, 3], ':', c='k')
        plt.legend(loc="best")
        plt.ylabel("Regression output")
        plt.xlabel("Input feature")
        plt.plot(self.X[:, 0], self.y, 'o', c='k')
        fig.savefig('interaction_feature/bin_combined.png')

    def bin_product(self):
        reg = LinearRegression().fit(self.X_product, self.y)

        line_product = np.hstack([self.line_binned, self.line * self.line_binned])
        fig = plt.figure()
        plt.plot(self.line, reg.predict(line_product), label='linear regression combined')

        for bin in self.bins:
            plt.plot([bin, bin], [-3, 3], ':', c='k')
        plt.legend(loc="best")
        plt.ylabel("Regression output")
        plt.xlabel("Input feature")
        plt.plot(self.X[:, 0], self.y, 'o', c='k')
        fig.savefig('interaction_feature/bin_product.png')
