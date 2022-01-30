import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class ModelBasedSelection:
    cancer = load_breast_cancer()

    rng = np.random.RandomState(42)
    noise = rng.normal(size=(len(cancer.data), 50))
    X_w_noise = np.hstack([cancer.data, noise])
    X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=5)
    select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
    select.fit(X_train, y_train)
    X_train_l1 = select.transform(X_train)
    X_test_l1 = select.transform(X_test)

    def main(self):
        self.show_selected_data()
        self.logistic_regression()

    def show_selected_data(self):
        mask = self.select.get_support()
        print("X_train.shape: {}".format(self.X_train.shape))
        print("X_train_selected.shape: {}".format(self.X_train_l1.shape))
        print(mask)

        plt.matshow(mask.reshape(1, -1), cmap='gray_r')
        plt.xlabel("Sample index")
        plt.show()

    def logistic_regression(self):
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        print("Score with all features: {:.3f}".format(lr.score(self.X_test, self.y_test)))

        lr.fit(self.X_train_l1, self.y_train)
        print("Score with selected features: {:.3f}".format(lr.score(self.X_test_l1, self.y_test)))
