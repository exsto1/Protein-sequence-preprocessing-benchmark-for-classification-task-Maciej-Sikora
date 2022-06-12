import pickle
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier


def K_neighbours():
    with open('../data/clean_dataset_compressed.pkl', 'rb') as f:
        dataset = pickle.load(f)

    model = KNeighborsClassifier()
    model.fit(dataset[0], dataset[1])

    prediction = model.predict(dataset[2])

    counter = 0
    for i in range(len(prediction)):
        if prediction[i] == dataset[3][i]:
            counter += 1

    print(counter / len(prediction) * 100)


if __name__ == '__main__':
    K_neighbours()