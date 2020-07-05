from typing import List, Optional
from dataclasses import dataclass

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


EOS = 'eos'


def add_eos(sentences: List[str]):
    return list(map(
        lambda s: s + ' ' + EOS,
        sentences
    ))


class TextParser:
    def __init__(
            self,
            num_words=None,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n、。',
            is_fixed_len=True,
    ):
        """語彙数を限定する場合はnum_wordsを指定する。系列長が固定長ならis_fix_lenをTrueにする"""
        self.tokenizer = Tokenizer(num_words, filters)
        self.max_len = None
        self.is_fixed_len = is_fixed_len
        self.eos_word = EOS

    def fit(
            self,
            texts: List[str],
            max_len: Optional[int] = None,
    ):
        """
        テキストにトークナイザーをfitさせる。固定長なら、max_lenで系列長を指定する。
        max_lenを指定しなければ、textsから取得する
        """
        if self.is_fixed_len:
            self.max_len = max_len
            if max_len is None:
                self.max_len = max(map(len, texts))
        self.tokenizer.fit_on_texts(texts)

    def texts_to_seqs(self, texts: List[str]) -> np.ndarray:
        seqs = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=self.max_len)

    def texts_to_one_hot(self, texts: List[str]) -> np.ndarray:
        seqs = self.texts_to_seqs(texts)
        return to_categorical(seqs[:, 1:])[:, :, 1:]

    def texts_to_mask(self, texts: List[str]) -> np.ndarray:
        seqs = self.texts_to_seqs(texts)
        return (seqs > 0).astype(int)

    def texts_to_pos(self, texts: List[str]) -> np.ndarray:
        mask = self.texts_to_mask(texts)
        return np.cumsum(mask, axis=1)

    def seqs_to_texts(self, seqs):
        return self.tokenizer.sequences_to_texts(seqs)

    @property
    def eos_index(self):
        return self.tokenizer.word_index[self.eos_word]


@dataclass
class TextData:
    texts: List[str]
    parser: TextParser

    @property
    def seqs(self):
        return self.parser.texts_to_seqs(self.texts)

    @property
    def mask(self):
        return self.parser.texts_to_mask(self.texts)

    @property
    def one_hot(self):
        return self.parser.texts_to_one_hot(self.texts)

    @property
    def pos(self):
        return self.parser.texts_to_pos(self.texts)


@dataclass(frozen=True)
class DataSet:
    train_origin: TextData
    test_origin: TextData
    train_trans: TextData
    test_trans: TextData
    origin_parser: TextParser
    trans_parser: TextParser


def make_data_set(
        train_origin_texts: List[str],
        train_trans_texts: List[str],
        test_origin_texts: Optional[List[str]],
        test_trans_texts: Optional[List[str]],
        origin_num_words: Optional[int] = None,
        trans_num_words: Optional[int] = None,
        is_fixed_len: bool = True,
        origin_max_len: Optional[int] = None,
        trans_max_len: Optional[int] = None,
        has_eos: bool = False,
):
    if not has_eos:
        train_origin_texts = add_eos(train_origin_texts)
        test_origin_texts = add_eos(test_origin_texts)
        train_trans_texts = add_eos(train_trans_texts)
        test_trans_texts = add_eos(test_trans_texts)
    origin_parser = TextParser(
        num_words=origin_num_words,
        is_fixed_len=is_fixed_len,
    )
    origin_parser.fit(train_origin_texts, max_len=origin_max_len)
    trans_parser = TextParser(
        num_words=trans_num_words,
        is_fixed_len=is_fixed_len,
    )
    trans_parser.fit(train_trans_texts, max_len=trans_max_len)
    return DataSet(
        train_origin=TextData(train_origin_texts, origin_parser),
        test_origin=TextData(test_origin_texts, origin_parser),
        train_trans=TextData(train_trans_texts, trans_parser),
        test_trans=TextData(test_trans_texts, trans_parser),
        origin_parser=origin_parser,
        trans_parser=trans_parser,
    )
