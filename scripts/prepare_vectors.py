from BIOVEC.source.models import ProtVec
from BIOVEC.source.data_load import DataHolder
import os
import pickle
import shutil
import numpy as np
from sklearn.model_selection import train_test_split


def split_data_to_classes(infile="../data/clean_dataset.pkl", output_folder="../data/vectors/class_folder", output_combined_file="../data/vectors/combined.fasta"):
    with open(infile, 'rb') as f:
        dataset = pickle.load(f)

    try:
        os.mkdir(output_folder)
    except:
        shutil.rmtree(output_folder)
        os.mkdir(output_folder)

    if os.path.exists(output_combined_file):
        os.remove(output_combined_file)

    fam_stat = {}
    for i in range(len(dataset[0])):
        if dataset[1][i] not in fam_stat:
            fam_stat[dataset[1][i]] = 1
        else:
            fam_stat[dataset[1][i]] += 1

    for i in range(len(dataset[2])):
        if dataset[3][i] not in fam_stat:
            fam_stat[dataset[3][i]] = 1
        else:
            fam_stat[dataset[3][i]] += 1

    min_size = min(list(fam_stat.values()))

    families = {}
    for i in range(len(dataset[0])):
        if dataset[1][i] not in families:
            families[dataset[1][i]] = [dataset[0][i]]
        else:
            if len(families[dataset[1][i]]) < min_size:
                families[dataset[1][i]].append(dataset[0][i])

    for i in range(len(dataset[2])):
        if dataset[3][i] not in families:
            families[dataset[3][i]] = [dataset[2][i]]
        else:
            if len(families[dataset[3][i]]) < min_size:
                families[dataset[3][i]].append(dataset[2][i])

    for i in families:
        with open(f"{output_folder}/{i}", "w") as outfile:
            for i1 in range(len(families[i])):
                outfile.write(f">{i}_{i1}\n")
                outfile.write(f"{families[i][i1]}\n")

    with open(output_combined_file, "w") as outfile:
        for i in families:
            for i1 in range(len(families[i])):
                outfile.write(f">{i}_{i1}\n")
                outfile.write(f"{families[i][i1]}\n")


def prepare_biovec_model(infile="../data/vectors/combined.fasta", outfile="../data/vectors/ProtVec_model.model", corpus="../data/vectors/combined_corpus.cor"):
    if os.path.exists(corpus):
        os.remove(corpus)
    if os.path.exists(outfile):
        os.remove(outfile)

    pv_model = ProtVec(infile)
    pv_model.save(outfile)


def load_biovec_model_with_classes(input_model="../data/vectors/ProtVec_model.model", class_folder="../data/vectors/class_folder", outfile="../data/clean_dataset_biovec.pkl", train_size="500"):
    model_fname = input_model
    class_filenames = os.listdir(class_folder)
    class_paths = [f"{class_folder}/{i}" for i in class_filenames]
    class_names = [(class_filenames[i], i) for i in range(len(class_filenames))]

    data = DataHolder.load_with_protvec(model_fname, class_paths, class_names)

    X = np.array([x.protvec_repr for x in data]).reshape(-1, 100).astype('float32')
    y = np.array([x.cls for x in data]).reshape(-1, 1).astype('int8')

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=int(train_size), stratify=y)

    newdata = [X_train, y_train, X_test, y_test]

    with open(outfile, 'wb') as f:
        pickle.dump(newdata, f)

    return data


if __name__ == '__main__':
    # split_data_to_classes()
    # prepare_biovec_model()
    load_biovec_model_with_classes()
    pass