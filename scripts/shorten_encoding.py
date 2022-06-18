import pickle
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from leven import levenshtein


def compress_protein_data_original(infile='./data/clean_dataset.pkl', outfile='./data/clean_dataset_original.pkl'):
    with open(infile, 'rb') as f:
        dataset = pickle.load(f)

    new_dataset = [[], [], [], []]

    for i in dataset[0]:
        new_dataset[0].append([i0 for i0 in i])

    for i in dataset[2]:
        new_dataset[2].append([i0 for i0 in i])

    temp_seq_sizes = [len(i) for i in new_dataset[0]]
    temp_seq_sizes.extend([len(i) for i in new_dataset[2]])
    max_size = max(temp_seq_sizes)

    for i in range(len(new_dataset[0])):
        temp_len = len(new_dataset[0][i])
        new_dataset[0][i].extend(["-" for i in range(max_size - temp_len)])

    for i in range(len(new_dataset[2])):
        temp_len = len(new_dataset[2][i])
        new_dataset[2][i].extend(["-" for i in range(max_size - temp_len)])

    new_dataset[0] = np.asarray(new_dataset[0])
    new_dataset[1] = np.asarray(dataset[1])
    new_dataset[2] = np.asarray(new_dataset[2])
    new_dataset[3] = np.asarray(dataset[3])

    # new_dataset[0] = dataset[0]
    # new_dataset[1] = np.asarray(dataset[1][:])
    # new_dataset[2] = dataset[2]
    # new_dataset[3] = np.asarray(dataset[3][:])


    with open(outfile, 'wb') as f:
        pickle.dump(new_dataset, f)


def compress_protein_data_singletons(infile='./data/clean_dataset.pkl', outfile='./data/clean_dataset_singletons.pkl'):
    with open(infile, 'rb') as f:
        dataset = pickle.load(f)

    new_dataset = [[], [], [], []]

    encoder_decoder = {}
    counter = 1
    for i in dataset[0]:
        for i1 in range(len(i)):
            frag = i[i1]
            if frag not in encoder_decoder:
                encoder_decoder[frag] = counter
                counter += 1

    for i in dataset[2]:
        for i1 in range(len(i)):
            frag = i[i1]
            if frag not in encoder_decoder:
                encoder_decoder[frag] = counter
                counter += 1

    for i in dataset[0]:
        temp = []
        for i1 in range(len(i)):
            frag = i[i1]
            temp.append(encoder_decoder[frag])

        temp = np.asarray(temp)
        temp = temp.astype("int8")
        new_dataset[0].append(temp)

    for i in dataset[2]:
        temp = []
        for i1 in range(len(i)):
            frag = i[i1]
            temp.append(encoder_decoder[frag])

        temp = np.asarray(temp)
        temp = temp.astype("int8")
        new_dataset[2].append(temp)


    temp_seq_sizes = [len(i) for i in new_dataset[0]]
    temp_seq_sizes.extend([len(i) for i in new_dataset[2]])
    max_size = max(temp_seq_sizes)

    for i in range(len(new_dataset[0])):
        temp_len = len(new_dataset[0][i])
        new_dataset[0][i] = np.pad(new_dataset[0][i], (0, max_size - temp_len))

    for i in range(len(new_dataset[2])):
        temp_len = len(new_dataset[2][i])
        new_dataset[2][i] = np.pad(new_dataset[2][i], (0, max_size - temp_len))

    new_dataset[0] = np.asarray(new_dataset[0], dtype="int8")
    new_dataset[1] = np.asarray(dataset[1][:])
    new_dataset[2] = np.asarray(new_dataset[2], dtype="int8")
    new_dataset[3] = np.asarray(dataset[3][:])

    with open(outfile, 'wb') as f:
        pickle.dump(new_dataset, f)


def compress_protein_data_triplets(infile='./data/clean_dataset.pkl', outfile='./data/clean_dataset_triplets.pkl'):
    with open(infile, 'rb') as f:
        dataset = pickle.load(f)

    frag_size = 3
    encoder_decoder = {}
    counter = 1
    for i in dataset[0]:
        for i1 in range(0, len(i), frag_size):
            frag = i[i1:i1+frag_size]
            if frag not in encoder_decoder:
                encoder_decoder[frag] = counter
                counter += 1

    for i in dataset[2]:
        for i1 in range(0, len(i), frag_size):
            frag = i[i1:i1+frag_size]
            if frag not in encoder_decoder:
                encoder_decoder[frag] = counter
                counter += 1

    new_dataset = [[], [], [], []]
    for i in dataset[0]:
        temp = []
        for i1 in range(0, len(i), frag_size):
            frag = i[i1:i1+frag_size]
            temp.append(encoder_decoder[frag])

        temp = np.asarray(temp)
        temp = temp.astype("int16")
        new_dataset[0].append(temp)

    for i in dataset[2]:
        temp = []
        for i1 in range(0, len(i), frag_size):
            frag = i[i1:i1+frag_size]
            temp.append(encoder_decoder[frag])

        temp = np.asarray(temp)
        temp = temp.astype("int16")
        new_dataset[2].append(temp)

    temp_seq_sizes = [len(i) for i in new_dataset[0]]
    temp_seq_sizes.extend([len(i) for i in new_dataset[2]])
    max_size = max(temp_seq_sizes)

    for i in range(len(new_dataset[0])):
        temp_len = len(new_dataset[0][i])
        new_dataset[0][i] = np.pad(new_dataset[0][i], (0, max_size-temp_len))

    for i in range(len(new_dataset[2])):
        temp_len = len(new_dataset[2][i])
        new_dataset[2][i] = np.pad(new_dataset[2][i], (0, max_size-temp_len))

    new_dataset[0] = np.asarray(new_dataset[0], dtype="int16")
    new_dataset[1] = np.asarray(dataset[1][:])
    new_dataset[2] = np.asarray(new_dataset[2], dtype="int16")
    new_dataset[3] = np.asarray(dataset[3][:])

    with open(outfile, 'wb') as f:
        pickle.dump(new_dataset, f)


def compress_protein_data_sum_of_triplets(infile='./data/clean_dataset.pkl', outfile='./data/clean_dataset_sum_triplets.pkl', n=-1):
    with open(infile, 'rb') as f:
        dataset = pickle.load(f)

    if n == -1:
        n = len(dataset[0]) / 20

    triplet_list = {}
    for i0 in [0, 2]:
        for i in tqdm(dataset[i0]):
            for i1 in range(0, len(i)):
                frag = i[i1:i1+3]
                if len(frag) == 3:
                    if frag not in triplet_list:
                        triplet_list[frag] = 1
                    else:
                        triplet_list[frag] += 1


    chosen = []
    for i in triplet_list:
        if triplet_list[i] > n:
            chosen.append(i)

    print(len(chosen))

    data0 = []
    for i in tqdm(dataset[0]):
        temp = []
        for i1 in range(len(chosen)):
            temp.append(i.count(chosen[i1]))
        temp = np.asarray(temp, dtype="int16")
        data0.append(temp)

    data0 = np.asarray(data0, dtype="int16")

    data2 = []
    for i in tqdm(dataset[2]):
        temp = []
        for i1 in range(len(chosen)):
            temp.append(i.count(chosen[i1]))
        temp = np.asarray(temp, dtype="int16")
        data2.append(temp)

    data2 = np.asarray(data2, dtype="int16")

    new_dataset = [data0, dataset[1], data2, dataset[3]]
    new_dataset[1] = np.asarray(dataset[1][:])
    new_dataset[3] = np.asarray(dataset[3][:])

    for i in new_dataset:
        print(i, len(i))

    with open(outfile, 'wb') as f:
        pickle.dump(new_dataset, f)


def compress_protein_data_sum_of_k_mers(infile='./data/clean_dataset.pkl', outfile='./data/clean_dataset_sum_k_mers.pkl', k=7, n=10, edit=1):
    with open(infile, 'rb') as f:
        dataset = pickle.load(f)

    kmer_list = {}
    for i0 in [0, 2]:
        for i in tqdm(dataset[i0]):
            for i1 in range(0, len(i)):
                frag = i[i1:i1+k]
                if len(frag) == k:
                    if frag not in kmer_list:
                        kmer_list[frag] = 1
                    else:
                        kmer_list[frag] += 1

    chosen0 = []
    for i in kmer_list:
        if kmer_list[i] > n:
            chosen0.append(i)

    chosen = {}
    for i in tqdm(chosen0):
        if i not in chosen:
            state = True
            for i1 in chosen:
                l = levenshtein(i, i1)
                if l <= edit:
                    chosen[i1].append(i)
                    state = False
                    break
            if state:
                chosen[i] = [i]

    for i in chosen:
        print(i, chosen[i])

    print(len(chosen))

    chosen_kmers = list(chosen.keys())
    data0 = []
    for i in tqdm(dataset[0]):
        temp = []
        for i1 in range(len(chosen_kmers)):
            t_count = 0
            for i2 in chosen[chosen_kmers[i1]]:
                t_count += i.count(i2)
            temp.append(t_count)
        temp = np.asarray(temp, dtype="int16")
        data0.append(temp)

    data0 = np.asarray(data0, dtype="int16")

    data2 = []
    for i in tqdm(dataset[2]):
        temp = []
        for i1 in range(len(chosen_kmers)):
            t_count = 0
            for i2 in chosen[chosen_kmers[i1]]:
                t_count += i.count(i2)
            temp.append(t_count)
        temp = np.asarray(temp, dtype="int16")
        data2.append(temp)

    data2 = np.asarray(data2, dtype="int16")

    new_dataset = [data0, dataset[1], data2, dataset[3]]
    new_dataset[1] = np.asarray(dataset[1][:])
    new_dataset[3] = np.asarray(dataset[3][:])

    with open(outfile, 'wb') as f:
        pickle.dump(new_dataset, f)


if __name__ == '__main__':
    # compress_protein_data_original(infile='../data/clean_dataset.pkl', outfile='../data/clean_dataset_original.pkl')
    # compress_protein_data_singletons(infile='../data/clean_dataset.pkl', outfile='../data/clean_dataset_original.pkl')
    # compress_protein_data_triplets(infile='../data/clean_dataset.pkl', outfile='../data/clean_dataset_original.pkl')
    # compress_protein_data_sum_of_triplets(infile='../data/clean_dataset.pkl', outfile='../data/clean_dataset_sum_triplets.pkl')
    compress_protein_data_sum_of_k_mers(infile='../data/clean_dataset.pkl', outfile='../data/clean_dataset_sum_k_mers.pkl', k=7, n=20, edit=2)
    pass
