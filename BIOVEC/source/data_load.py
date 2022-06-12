from typing import Iterable, List, Tuple, Union

import numpy as np
from gensim.models.word2vec import Word2Vec
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from sklearn.preprocessing import StandardScaler

from BIOVEC.source.sequence import Sequence


class DataHolder:
    def __init__(self, data: List[Sequence]) -> None:
        self.data = list(data)
        self.data_desc = set(x.dataset for x in self)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, ind) -> Union[Sequence, "DataHolder"]:
        """Index a DataHolder object with list of indices, same as numpy"""
        if isinstance(ind, int):
            return self.data[ind]
        elif isinstance(ind, slice):
            return DataHolder(self.data[ind])
        elif isinstance(ind, list):
            try:
                return DataHolder([self[indice] for indice in ind])
            except IndexError:
                print("No such index")
            except TypeError:
                print("Indices must be integers")
        elif isinstance(ind, np.ndarray):
            ind = ind.flatten().tolist()
            return self[ind]
        else:
            raise TypeError

    def __iter__(self):
        yield from self.data

    def __getattr__(self, item) -> "DataHolder":
        if item in self.data_desc:
            return DataHolder([x for x in self if x.dataset == item])
        else:
            raise AttributeError

    def __str__(self) -> str:
        return """DataHolder with {} sequences""".format(len(self))

    def __getstate__(self):
        """Needed for pickling to work"""
        return {"data": self.data, "data_desc": self.data_desc}

    def __setstate__(self, state):
        """Needed for pickling to work"""
        self.data = state["data"]
        self.data_desc = state["data_desc"]

    def standarize_vecs(self) -> None:
        """Standarize vectors in self.data (0 mean, 1 variance)"""
        scaler = StandardScaler()
        scaler.fit([x.protvec_repr for x in self])
        for elem in self:
            elem.protvec_repr = scaler.transform(elem.protvec_repr.reshape(1, -1))

    def replace_label(self, dataset_name: str, target_label: int) -> None:
        """Replace label in samples coming from specified dataset with target label"""
        for sample in self:
            if sample.dataset == dataset_name:
                sample.cls = target_label

    def save(self, fname):
        import pickle

        with open(fname, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, fname) -> "DataHolder":
        import pickle

        with open(fname, "rb"):
            return pickle.load(fname)

    @classmethod
    def load_with_protvec(
        cls,
        model_fname: str,
        fasta_filenames: Iterable[str],
        dataset_desc_and_class: Iterable[Tuple[str, int]],
    ) -> "DataHolder":
        """Embedd vectors from fasta files using specified model
            Arguments:
                model_fname: filename of a saved ProtVec model object
                fasta_filenames: an iterable of fasta filenames
                                 from which protein sequences are going to be embedded
                dataset_desc_and_class: an iterable of tuples of a form
                                        (dataset_description, class_assingment),
                                        needs to have the same length as fasta_filenames
            Return:
                A DataHolder object of Sequences
            """
        protvec_model = Word2Vec.load(model_fname)
        data = []
        for fasta_fname, (dataset_desc, _cls) in zip(
            fasta_filenames, dataset_desc_and_class
        ):
            for record in sequence_loader(fasta_fname):
                rec = Sequence(record, protvec_model.to_vecs(str(record.seq)))
                rec.cls = _cls
                rec.dataset = dataset_desc
                data.append(rec)
        return cls(data)

    @classmethod
    def load_with_bow(
        cls,
        model_fname: str,
        fasta_filenames: Iterable[str],
        dataset_desc_and_class: Iterable[Tuple[str, int]],
    ) -> "DataHolder":
        """Transform sequences to bag-of-words representation given a model,
           return a DataHolder object with them"""
        data = []
        import pickle

        with open(model_fname, "rb") as file:
            model = pickle.load(file)
        for fasta_fname, (dataset_desc, _cls) in zip(
            fasta_filenames, dataset_desc_and_class
        ):
            for record in sequence_loader(fasta_fname):
                rec = Sequence(record)
                model.transform(rec)
                rec.cls = _cls
                rec.dataset = dataset_desc
                data.append(rec)
        return cls(data)


def sequence_loader(fasta_fname: str) -> Iterable[SeqRecord]:
    """Open a fasta file and yield SeqRecord objects"""
    with open(fasta_fname, "r") as handle:
        yield from SeqIO.parse(handle, "fasta")
