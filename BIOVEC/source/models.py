import numpy as np
import warnings
import os
from typing import Union, List, Iterator, Iterable

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer

from BIOVEC.source.sequence import Sequence


class NgramNotFoundWarning(UserWarning):
    pass


def split_sequence(sequence: str, n=3) -> Iterator[List[str]]:
    """Split sequence to n lists of n-grams and yield them."""
    for j in range(n):
        yield [sequence[i : i + n] for i in range(j, len(sequence) - n + 1, n)]


def generate_corpus(fasta_fname: str, corpus_fname: str, n=3):
    """
    Save a corpus file generated from fasta file with path fasta_fname.

    Args:
        fasta_fname: a path to source fasta file containing aminoacid sequences to be iterated on
        corpus_fname: a path to corpus file to be saved
        n: "n" in "n-gram"

    Returns:
        None
    """
    with open(corpus_fname, "w") as corpus_file:
        for list_of_ngrams in generate_ngrams(fasta_fname, n):
            corpus_file.write(" ".join(list_of_ngrams))
            corpus_file.write("\n")


def generate_ngrams(fasta_fname: str, n=3) -> List[List[str]]:
    """
    Yield n lists of n-grams for each sequence in fasta file.

    Args:
        fasta_fname: a source fasta filename containing aminoacid sequences to be iterated on
        n: "n" in "n-gram"

    Yield:
        lists of n-grams
    """
    with open(fasta_fname, "r") as handle:
        for record in SeqIO.parse(handle, format="fasta"):
            yield from split_sequence(str(record.seq), n=n)


class ProtVec(Word2Vec):
    """
        Inspired by biovec package by kyu99: https://github.com/kyu999/biovec
        Algorithm comes from Ehsaneddin Asgari, Mohammad R.K. Mofrad, 2015

        ---

        A subclass of gensim.models.Word2Vec

    """

    def __init__(
        self,
        filename: str,
        corpus=False,
        n=3,
        size=100,
        sg=0,
        window=25,
        min_count=1,
        workers=4,
        **kwargs
    ) -> None:
        """
        Initialize ProtVec object.

        Args:
            filename: if corpus==True it should be a path to a corpus file, if not to fasta file
            corpus: if True a corpus is read from file, if False a corpus is generated from a fasta file named filename
        Args needed for Word2Vec:
            n: "n" in "n-gram"
            size: size of word2vec vectors
            sg: training algorithm; 0 for CBOW, 1 for skip-gram
            window: maximum distance between the current and predicted word within a sentence
            min_count: ignores all words with total frequency lower than this
            workers: how many worker cores to use training a model
            kwargs: optional keyword arguments
        """

        self.n = n
        self.filename = filename
        self.size = size
        if corpus:
            sentences = LineSentence(filename)
        elif os.path.isfile("".join([filename[:-6], "_corpus.cor"])):
            sentences = LineSentence("".join([filename[:-6], "_corpus.cor"]))
        else:
            generate_corpus(
                fasta_fname=filename,
                corpus_fname="".join([filename[:-6], "_corpus.cor"]),
                n=n,
            )
            sentences = LineSentence("".join([filename[:-6], "_corpus.cor"]))

        super().__init__(
            sentences=sentences,
            size=size,
            sg=sg,
            window=window,
            min_count=min_count,
            workers=workers,
            **kwargs
        )

    def to_vecs(self, seq: str, sum_vecs=True) -> Union[np.ndarray, List[np.ndarray]]:
        vecs = []
        for n_grams in split_sequence(seq, n=self.n):
            cur_vec = np.zeros(self.size)
            for n_gram in n_grams:
                try:
                    cur_vec = cur_vec + self.wv.__getitem__(n_gram)
                except KeyError:
                    warnings.warn(
                        "Model hasn't trained on this n-gram: {}".format(n_gram),
                        NgramNotFoundWarning,
                    )
            vecs.append(cur_vec)
        if sum_vecs:
            return np.sum(vecs, axis=0)
        else:
            return vecs


class BagOfWords:
    def __init__(self, fasta_fname: str, n: int, **kwargs):
        self.fasta_fname = fasta_fname
        self.n = n
        self.model = CountVectorizer(**kwargs)
        if os.path.isfile("".join([fasta_fname[:-6], str(n), "_corpus.cor"])):
            corpus_filename = "".join([fasta_fname[:-6], str(n), "_corpus.cor"])
        else:
            generate_corpus(
                fasta_fname, "".join([fasta_fname[:-6], str(n), "_corpus.cor"]), n
            )
            corpus_filename = "".join([fasta_fname[:-6], str(n), "_corpus.cor"])
        with open(corpus_filename, "r") as file:
            self.model.fit(file)

    def transform_new(
        self, data: Union[Sequence, Iterable[Sequence]]
    ) -> Union[Sequence, Iterable[Sequence]]:
        cp_l = list(data)
        self.transform(cp_l)
        return cp_l

    def transform(self, data: Iterable[Sequence]):
        if isinstance(data, Sequence):
            data.bag_of_words_repr = np.array(
                np.concatenate(
                    [
                        self.model.transform([" ".join(seq)]).todense()
                        for seq in split_sequence(str(data.seq), self.n)
                    ],
                    axis=1,
                )
            )
        else:
            for sequence in data:
                sequence.bag_of_words_repr = np.array(
                    np.concatenate(
                        [
                            self.model.transform([" ".join(seq)]).todense()
                            for seq in split_sequence(str(sequence.seq), self.n)
                        ],
                        axis=1,
                    )
                )

    def save(self, filename):
        import pickle

        with open(filename, "wb") as file:
            pickle.dump(self, file)
