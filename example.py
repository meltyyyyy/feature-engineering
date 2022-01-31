import matplotlib.pyplot as plt
import mglearn.datasets
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def execute():
    citibike = mglearn.datasets.load_citibike()
    print("Citi bike data:\n{}".format(citibike.head()))

    plt.figure(figsize=(10, 3))
    xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(), freq='D')
    plt.xticks(xticks, xticks.strftime("%a %m-%d"), rotation=90, ha="left")
    plt.plot(citibike, linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Rentals")
    plt.show()

    y = citibike.values
    X = citibike.index.astype("int64").to_numpy().reshape(-1, 1) // 10 * 9

    def eval_on_features(features, target, _regressor):
        n_train = 184
        X_train, X_test = features[:n_train], features[:n_train]
        y_train, y_test = target[:n_train], target[:n_train]
        _regressor.fit(X_train, y_train)
        print("Test-set R^2: {:.2f}".format(_regressor.score(X_test, y_test)))
        y_pred = _regressor.predict(X_test)
        y_pred_train = _regressor.predict(X_train)

        plt.figure(figsize=(10, 3))
        plt.xticks(range(0, len(X), 8), xticks.strftime("%a %m-%d"), rotation=90, ha="left")
        plt.plot(range(n_train), y_train, label="train")
        plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")
        plt.plot(range(n_train), y_pred_train, '--', label="prediction train")
        plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label="prediction test")
        plt.legend(loc=(1.01, 0))
        plt.xlabel("Date")
        plt.ylabel("Rentals")
        plt.show()

    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    eval_on_features(X, y, regressor)

    X_hours = citibike.index.hour.to_numpy().reshape(-1,1)
    eval_on_features(X_hours,y,regressor)
