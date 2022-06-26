from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import numpy as np
import pandas as pd


def random_tree_func(datafile="../../data/clean_dataset_sum_triplets.pkl"):
    with open(datafile, 'rb') as f:
        data = pickle.load(f)

    params = {"random_state": [1234],
              "max_depth": [10, 20, 30, 40, 50],
              "max_features": [10, 15, 20]}
    model_combined = GridSearchCV(RandomForestClassifier(), param_grid=params, scoring="accuracy", cv=2, n_jobs=2, verbose=1, refit=True)
    try:
        if data[0].dtype.name != "str32":
            model_combined.fit(data[0], data[1])
        else:
            data[0] = pd.DataFrame(data[0])
            data[0] = data[0].apply(lambda i: pd.factorize(i)[0])
            data[2] = pd.DataFrame(data[2])
            data[2] = data[2].apply(lambda i: pd.factorize(i)[0])
            model_combined.fit(data[0], data[1])
    except:
        data[0] = pd.DataFrame(data[0])
        data[0] = data[0].apply(lambda i: pd.factorize(i)[0])
        data[2] = pd.DataFrame(data[2])
        data[2] = data[2].apply(lambda i: pd.factorize(i)[0])
        model_combined.fit(data[0], data[1])

    prediction = model_combined.predict(data[2])
    data[3] = data[3].flatten()
    real = data[3].tolist()

    correct = 0
    for i in range(len(prediction)):
        if prediction[i] == real[i]:
            correct += 1
    accuracy = correct / len(prediction)

    runtime = round(model_combined.refit_time_, 4)

    print("Best model:")
    for name in params:
        print(f"- {name}: {str(model_combined.best_params_[name])}")
    print(f"----- Model accuracy: {round(accuracy, 3)}")

    return round(accuracy, 4), runtime


if __name__ == '__main__':
    random_tree_func()

