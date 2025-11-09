import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bpe_tokenizer import BPETokenizer


def test_train_and_encode_decode_roundtrip():
    corpus = [
        "the quick brown fox",
        "the quick blue hare",
    ]
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=30)

    encoded = tokenizer.encode("the quick brown fox")
    decoded = tokenizer.decode(encoded)

    assert decoded == "the quick brown fox"


def test_raises_when_not_trained():
    tokenizer = BPETokenizer()
    with pytest.raises(RuntimeError):
        tokenizer.encode("hello")
    with pytest.raises(RuntimeError):
        tokenizer.decode([0])


def test_unknown_token_handling():
    corpus = ["hello world"]
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=10)

    encoded = tokenizer.encode("unknown token")
    unk_id = tokenizer.vocab[tokenizer.unk_token]
    assert unk_id in encoded
