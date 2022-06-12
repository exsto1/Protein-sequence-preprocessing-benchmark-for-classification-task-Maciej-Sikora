import numpy as np

from Bio import SeqRecord


class Sequence:
    def __init__(
        self,
        record: SeqRecord,
        protvec_repr: np.ndarray = np.empty(1),
        bag_of_words_repr: np.ndarray = np.empty(1),
    ):
        self.record = record
        self.seq = str(record.seq)
        self.protvec_repr = protvec_repr
        self.cls = ""
        self.dataset = ""
        self.bag_of_words_repr = bag_of_words_repr

    def __getattr__(self, item):
        """Give access to all SeqRecord attributes"""
        return self.record.__dict__[item]

    def __str__(self):
        return "<Sequence object at {}\nRecord name: {}\nClass: {}, Dataset: {}>".format(
            hex(id(self)), self.record.description, self.cls, self.dataset
        )

    def __getstate__(self):
        """Needed for pickling to work"""
        return {
            "record": self.record,
            "seq": self.seq,
            "protvec_repr": self.protvec_repr,
            "cls": self.cls,
            "dataset": self.dataset,
            "bag_of_words_repr": self.bag_of_words_repr,
        }

    def __setstate__(self, state):
        """Needed for pickling to work"""
        self.record = state["record"]
        self.seq = state["seq"]
        self.protvec_repr = state["protvec_repr"]
        self.cls = state["cls"]
        self.dataset = state["dataset"]
        self.bag_of_words_repr = state["bag_of_words_repr"]
