import os.path

import mglearn as mglearn
import pandas as pd
from IPython.core.display_functions import display
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class OneHotEncoding:
    _adult_path = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")
    _data = pd.read_csv(
        _adult_path, header=None, index_col=False,
        names=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status',
               'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country', 'income'
               ])
    _data = _data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']]

    def main(self):
        display(self._data.head())
        self.get_dummy_data()

    def get_dummy_data(self):
        print("Original features:\n", list(self._data), "\n")
        data_dummies = pd.get_dummies(self._data)
        print("Features after get_dummies:\n", list(data_dummies.columns))
        display(data_dummies.head())
        self.logistic_regression(data_dummies)

    def logistic_regression(self, data_dummies):
        features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
        X = features.values
        y = data_dummies['income_ >50K'].values
        print("X.shape: {} y.shape: {}".format(X.shape, y.shape))

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        logreg = LogisticRegression(max_iter=10000)
        logreg.fit(X_train, y_train)
        print("Test score: {:.2f}".format(logreg.score(X_test, y_test)))
