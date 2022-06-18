import shutil
import os
import pickle
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import hamming


def CLANS_seq_prepare(input_file="../data/clean_dataset_singletons.pkl", out_file="../data/CLANS/data_file_for_clans.fasta", clans_position="../CLANS/clans.jar"):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    data0 = list(data[0])
    data1 = list(data[1])

    data_all = {}
    for i in range(len(data[0])):
        if data1[i] in data_all:
            data_all[data1[i]].append(list(data0[i]))
        else:
            data_all[data1[i]] = [list(data0[i])]

    data = []
    for i in data_all:
        temp = []
        for i1 in data_all:
            x = pairwise_distances(data_all[i], data_all[i1])
            temp.append(x.mean())
        data.append(temp)

    for i in data:
        print(i)


if __name__ == '__main__':
    CLANS_seq_prepare()


