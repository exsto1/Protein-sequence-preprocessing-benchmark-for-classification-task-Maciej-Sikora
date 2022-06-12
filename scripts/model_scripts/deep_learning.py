import tensorflow as tf
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pickle
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def deep_learning_func(datafile="../../data/clean_dataset_triplets.pkl"):
    with open(datafile, 'rb') as f:
        data = pickle.load(f)

    datax = pd.DataFrame(data[0])
    datay = pd.DataFrame(data[1])
    datay = datay.apply(lambda x: pd.factorize(x)[0])

    print(len(datax), len(datay))

    model = Sequential()
    model.add(Dense(32, input_dim=628, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    model.fit(datax, datay)

    prediction = model.predict(data[2])
    correct = (prediction == data[3])
    accuracy = correct.sum() / correct.size

    print("Best model:")
    print(f"----- Model accuracy: {round(accuracy, 3)}")

if __name__ == '__main__':
    deep_learning_func()