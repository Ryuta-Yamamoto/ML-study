from dataclasses import dataclass
from typing import List
from pathlib import Path


dir_path = Path(__file__).parent / 'data'


@dataclass(frozen=True)
class TranslationDataSet:
    """翻訳タスクの原文と訳文対"""
    origin: List[str]
    trans: List[str]


def tanaka_corpus(is_train=True, en2ja=True):
    """
    田中コーパスをロードする。
    :param is_train 訓練データの場合はTrue, テストデータの場合はFalse
    :param en2ja 英->日翻訳ならTure, 日->英翻訳ならFalse
    """
    tanaka_dir = dir_path / 'tanaka_corpus/small_parallel_enja'
    train_en = tanaka_dir / 'train.en'
    train_ja = tanaka_dir / 'train.ja'
    test_en = tanaka_dir / 'test.en'
    test_ja = tanaka_dir / 'test.ja'

    if is_train:
        en_path = train_en
        ja_path = train_ja
    else:
        en_path = test_en
        ja_path = test_ja

    with open(en_path) as f:
        en = f.readlines()
    with open(ja_path) as f:
        ja = f.readlines()

    if en2ja:
        return TranslationDataSet(origin=en, trans=ja)
    return TranslationDataSet(origin=ja, trans=en)
