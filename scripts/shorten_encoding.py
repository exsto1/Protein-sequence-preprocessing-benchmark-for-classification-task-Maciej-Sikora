import pickle
import sys
import numpy as np


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


if __name__ == '__main__':
    compress_protein_data_original(infile='../data/clean_dataset.pkl', outfile='../data/clean_dataset_original.pkl')
    # compress_protein_data_singletons(infile='../data/clean_dataset.pkl', outfile='../data/clean_dataset_original.pkl')
    # compress_protein_data_triplets(infile='../data/clean_dataset.pkl', outfile='../data/clean_dataset_original.pkl')
