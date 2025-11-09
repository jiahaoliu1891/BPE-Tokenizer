# BPE-Tokenizer

A minimal implementation of a Byte Pair Encoding (BPE) tokenizer written from scratch.

## Usage

```python
from bpe_tokenizer import BPETokenizer

corpus = [
    "the quick brown fox",
    "the quick blue hare",
]

# Train the tokenizer
bpe = BPETokenizer()
bpe.train(corpus, vocab_size=30)

# Encode and decode
encoded = bpe.encode("the quick brown fox")
decoded = bpe.decode(encoded)

print(encoded)
print(decoded)
```

## Development

Install the test dependencies and run the test suite with `pytest`:

```bash
pip install -r requirements-dev.txt
pytest
```
