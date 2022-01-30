import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class UnivariateStatistics:
    cancer = load_breast_cancer()

    rng = np.random.RandomState(42)
    noise = rng.normal(size=(len(cancer.data), 50))
    X_w_noise = np.hstack([cancer.data, noise])
    X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=5)
    select = SelectPercentile(percentile=50)
    select.fit(X_train, y_train)
    X_train_selected = select.transform(X_train)
    X_test_selected = select.transform(X_test)

    def main(self):
        self.show_selected_data()
        self.logistic_regression()

    def show_selected_data(self):
        mask = self.select.get_support()
        print("X_train.shape: {}".format(self.X_train.shape))
        print("X_train_selected.shape: {}".format(self.X_train_selected.shape))
        print(mask)

        fig = plt.figure()
        plt.matshow(mask.reshape(1,-1), cmap='gray_r')
        plt.xlabel("Sample index")
        fig.savefig('feature_selection/univariate_statistics/selected_feature.png')

    def logistic_regression(self):
        lr = LinearRegression()
        lr.fit(self.X_train,self.y_train)
        print("Score with all features: {:.3f}".format(lr.score(self.X_test,self.y_test)))

        lr.fit(self.X_train_selected,self.y_train)
        print("Score with selected features: {:.3f}".format(lr.score(self.X_test_selected, self.y_test)))

