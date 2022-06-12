import tensorflow as tf
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import pickle
import numpy as np
import warnings
from itertools import product
from tqdm import tqdm
warnings.simplefilter(action='ignore', category=FutureWarning)


def deep_learning_func(datafile="../../data/clean_dataset_triplets.pkl"):
    with open(datafile, 'rb') as f:
        data = pickle.load(f)

    if data[0].dtype.name != "str32":
        datax = pd.DataFrame(data[0])
        datay = pd.DataFrame(data[1])
        datax_test = pd.DataFrame(data[2])
        datay_test = pd.DataFrame(data[3])
        x = datay.nunique()
    else:
        data[0] = pd.DataFrame(data[0])
        datax = data[0].apply(lambda i: pd.factorize(i)[0])
        data[2] = pd.DataFrame(data[2])
        datax_test = data[2].apply(lambda i: pd.factorize(i)[0])
        datay = pd.DataFrame(data[1])
        datay_test = pd.DataFrame(data[3])

    datay = to_categorical(datay)
    unique = list(datay.shape)[1]

    datay_test = to_categorical(datay_test)

    layers = [2, 3]
    nodes = [16, 64, 512]
    best_acc = 0
    for layer_n in layers:
        temp = [nodes for i in range(layer_n)]
        layer_possibilities = list(product(*temp))
        for layer_v in tqdm(layer_possibilities):
            model = Sequential()
            model.add(Dense(64, input_dim=datax.shape[1], activation='relu'))
            for layer_i in layer_v:
                model.add(Dense(layer_i, activation='relu'))
            model.add(Dense(unique, activation='sigmoid'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(datax, datay, verbose=0)
            _, accuracy = model.evaluate(datax_test, datay_test, verbose=0)
            if accuracy > best_acc:
                best_acc = accuracy
                print(best_acc)

    print(f"----- Model accuracy: {round(best_acc, 3)}")

    return round(best_acc, 4)


if __name__ == '__main__':
    deep_learning_func(datafile="../../data/clean_dataset_original.pkl")