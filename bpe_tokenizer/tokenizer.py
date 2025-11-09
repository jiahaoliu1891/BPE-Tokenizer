"""A minimal Byte Pair Encoding (BPE) tokenizer implementation."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

Pair = Tuple[str, str]
Word = Tuple[str, ...]


@dataclass
class BPETokenizer:
    """Train and apply a Byte Pair Encoding tokenizer.

    Parameters
    ----------
    unk_token:
        Token used when an input piece is not present in the vocabulary.
    """

    unk_token: str = "<unk>"
    merges: List[Pair] = field(default_factory=list)
    vocab: Dict[str, int] = field(default_factory=dict)

    def train(self, corpus: Iterable[str], vocab_size: int) -> None:
        """Train the tokenizer on the provided corpus.

        Parameters
        ----------
        corpus:
            Iterable with each element representing a text sample.
        vocab_size:
            Target vocabulary size (including ``unk_token``).
        """

        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")

        word_freqs = Counter()
        for line in corpus:
            words = [f"▁{w}" for w in line.strip().split() if w]
            if not words:
                continue
            for word in words:
                tokens = tuple(list(word) + ["</w>"])
                word_freqs[tokens] += 1

        if not word_freqs:
            raise ValueError("Corpus is empty and cannot be used for training")

        self.merges.clear()

        vocab_symbols = {self.unk_token}
        for word in word_freqs:
            vocab_symbols.update(symbol for symbol in word if symbol != "</w>")

        while len(vocab_symbols) < vocab_size:
            stats = self._get_stats(word_freqs)
            if not stats:
                break
            best = max(stats, key=stats.get)
            self.merges.append(best)
            word_freqs = self._merge_vocab(best, word_freqs)
            merged_symbol = "".join(best)
            vocab_symbols.add(merged_symbol)

        ordered_vocab = [self.unk_token]
        ordered_vocab.extend(sorted(vocab_symbols - {self.unk_token}))

        self.vocab = {token: idx for idx, token in enumerate(ordered_vocab)}

    def encode(self, text: str) -> List[int]:
        """Encode a text string into token ids."""

        if not self.vocab:
            raise RuntimeError("Tokenizer has not been trained")

        token_ids: List[int] = []
        for word in [f"▁{w}" for w in text.strip().split() if w]:
            symbols = list(word) + ["</w>"]
            for pair in self.merges:
                symbols = self._apply_merge(symbols, pair)
            for symbol in symbols:
                if symbol == "</w>":
                    continue
                clean_symbol = symbol[:-4] if symbol.endswith("</w>") else symbol
                token_ids.append(self.vocab.get(clean_symbol, self.vocab[self.unk_token]))
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode a sequence of token ids back to text."""

        if not self.vocab:
            raise RuntimeError("Tokenizer has not been trained")

        inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        pieces = []
        for idx in token_ids:
            token = inverse_vocab.get(idx, self.unk_token)
            pieces.append(token)
        text = "".join(pieces)
        return text.replace("▁", " ").strip()

    def _get_stats(self, word_freqs: Counter[Word]) -> Dict[Pair, int]:
        stats: Dict[Pair, int] = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if "</w>" in pair:
                    # Do not merge across the end of a word.
                    continue
                stats[pair] += freq
        return stats

    def _merge_vocab(self, pair: Pair, word_freqs: Counter[Word]) -> Counter[Word]:
        merged_vocab = Counter()
        bigram = "".join(pair)
        for word, freq in word_freqs.items():
            merged_word: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    merged_word.append(bigram)
                    i += 2
                else:
                    merged_word.append(word[i])
                    i += 1
            merged_vocab[tuple(merged_word)] += freq
        return merged_vocab

    def _apply_merge(self, symbols: List[str], pair: Pair) -> List[str]:
        merged: List[str] = []
        i = 0
        bigram = "".join(pair)
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                merged.append(bigram)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged
