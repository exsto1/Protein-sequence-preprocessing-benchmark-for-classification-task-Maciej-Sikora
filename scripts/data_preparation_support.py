import random

import numpy as np
from random import shuffle
import pickle


def data_preparation(file_path="../data/full/uniprot-reviewed_yes.tab", outfile_path="../data/data_file.fasta", n_org=3000, n_fam=50):
    org_stats = {}
    with open(file_path) as fileh:
        state = True
        while state:
            line = fileh.readline()
            if line:
                line = line.rstrip().split("\t")
                org = line[2]
                if org in org_stats:
                    org_stats[org] += 1
                else:
                    org_stats[org] = 1
            else:
                state = False

    org_names = list(org_stats.keys())
    org_names.sort(key=lambda i: org_stats[i], reverse=True)

    org_list = org_names[:n_org]

    data = []

    with open(file_path) as fileh:
        state = True
        while state:
            line = fileh.readline()
            if line:
                line = line.rstrip().split("\t")
                org = line[2]
                if org in org_list:
                    fam = line[1].split(";")
                    if len(fam) == 2:
                        if 50 < len(line[-1]) < 2000:
                            data.append(line)
            else:
                state = False


    family_stats = {}
    for i in range(len(data)):
        data[i][1] = data[i][1].rstrip(";")
        fam_t = data[i][1]
        if fam_t in family_stats:
            family_stats[fam_t] += 1
        else:
            family_stats[fam_t] = 1

    fam_names = list(family_stats.keys())
    fam_names.sort(key=lambda i: family_stats[i], reverse=True)

    fam_list = fam_names[:n_fam]

    with open(outfile_path, "w") as outfile:
        for i in range(len(data)):
            if data[i][1] in fam_list:
                outfile.write(f">{data[i][0]}_{data[i][1]}_{data[i][2]}\n")
                outfile.write(f"{data[i][-1]}\n")

    print("Done!")


def split_to_train_test(train_val=400, infile="../data/clustering/data_file_clustered.fasta", outfile="../data/clean_dataset.pkl"):
    with open(infile) as fileh:
        file = fileh.readlines()

    file = [i.rstrip() for i in file]

    seq = [[file[i], file[i+1]] for i in range(0, len(file), 2)]

    families = {}
    for i in range(len(seq)):
        temp_fam = seq[i][0].split("_")[1]
        if temp_fam in families:
            families[temp_fam].append(seq[i])
        else:
            families[temp_fam] = [seq[i]]

    filtered_families = {i: families[i] for i in families if len(families[i]) > train_val}
    min_size = min([len(i) for i in list(filtered_families.values())])

    # [X_train, y_train, X_test, y_test]
    data = [[], [], [], []]

    counter = 0
    for i in families:
        counter += 1
        shuffle(families[i])

        if len(families[i]) >= train_val:
            for i1 in range(len(families[i])):
                if i1 < train_val:
                    data[0].append(families[i][i1][1])
                    data[1].append(counter)
                else:
                    if i1 < min_size:
                        data[2].append(families[i][i1][1])
                        data[3].append(counter)

    indices1 = list(range(len(data[0])))
    random.shuffle(indices1)

    indices2 = list(range(len(data[2])))
    random.shuffle(indices2)

    new_data = [[], [], [], []]
    for i in indices1:
        new_data[0].append(data[0][i])
        new_data[1].append(data[1][i])

    for i in indices2:
        new_data[2].append(data[2][i])
        new_data[3].append(data[3][i])


    with open(outfile, 'wb') as f:
        pickle.dump(new_data, f)

    return train_val


if __name__ == '__main__':
    # data_preparation()
    split_to_train_test()
