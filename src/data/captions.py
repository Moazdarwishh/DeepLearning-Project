from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable

from .datasets import split_captions

_NON_LETTERS = re.compile(r"[^a-z\s]")
_APOSTROPHES = re.compile("['\u2019`\ufffd]")

PROTECTED_WORDS = {
    "bicolored", "tricolored", "multicolored", "otherside", "birdy", "wingbar", "wingbars",
    "eyering", "eyebrow", "eyestripe", "superciliary", "malar", "auriculars", "rectrices",
    "remiges", "tarsus", "tarsi", "nape", "lores", "culmen", "gorget",
}
_SPACES = re.compile(r"\s+")

COLOR_WORDS = [
    "black", "white", "brown", "grey", "gray", "yellow", "red", "blue", "orange",
    "green", "tan", "buff", "olive", "pink", "purple", "rufous", "iridescent", "cream",
]
STOPWORDS = {
    "a", "an", "the", "and", "with", "of", "is", "has", "this", "bird", "its", "on",
    "in", "that", "are", "it", "to", "as", "for", "small", "very", "which", "at", "be",
    "from", "by", "but", "or", "some", "while", "also", "have", "their", "very",
}


def normalize(text: str) -> str:
    """Lowercase, ASCII-fold, drop punctuation/digits, collapse whitespace."""
    text = _APOSTROPHES.sub("", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = _NON_LETTERS.sub(" ", text.lower())
    return _SPACES.sub(" ", text).strip()


def clean_caption(text: str, spell_fixes: dict[str, str] | None = None) -> str:
    """Clean one caption; spell fixes may split a token into two words."""
    tokens = normalize(text).split()
    if spell_fixes:
        tokens = [spell_fixes.get(t, t) for t in tokens]
    return " ".join(tokens)


def _split_merged(word: str, word_counts: Counter, bigram_counts: Counter | None,
                  min_count: int, min_bigram: int = 5) -> str | None:
    best = None
    for k in range(1, len(word)):
        left, right = word[:k], word[k:]
        if (len(left) == 1 and left not in {"a", "i"}) or len(right) < 2:
            continue
        lc, rc = word_counts.get(left, 0), word_counts.get(right, 0)
        bc = bigram_counts.get((left, right), 0) if bigram_counts is not None else min(lc, rc)
        if lc >= min_count and rc >= min_count and bc >= min_bigram:
            if best is None or bc > best[0]:
                best = (bc, f"{left} {right}")
    return best[1] if best else None


def build_spell_fixes(word_counts: Counter, bigram_counts: Counter | None = None, max_rare_count: int = 3,
                      min_target_count: int = 20, min_len: int = 4) -> dict[str, str]:
    """Map rare unknown words to a frequent corpus word (conservative typo fixing)."""
    from spellchecker import SpellChecker

    spell = SpellChecker()
    candidates = [w for w, c in word_counts.items()
                  if c <= max_rare_count and len(w) >= min_len and w not in PROTECTED_WORDS]
    unknown = spell.unknown(candidates)
    fixes: dict[str, str] = {}
    for word in sorted(unknown):
        merged = _split_merged(word, word_counts, bigram_counts, min_target_count)
        if merged:
            fixes[word] = merged
            continue
        correction = spell.correction(word)
        if correction and correction != word and word_counts.get(correction, 0) >= min_target_count:
            fixes[word] = correction
    return fixes


def caption_statistics(descriptions: Iterable[str], top_k: int = 30) -> dict:
    """Raw-caption statistics for the EDA (computed before cleaning)."""
    per_image_counts: Counter = Counter()
    lengths: list[int] = []
    word_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    raw_chars: Counter = Counter()
    non_ascii_captions = empty_lines = duplicate_within_image = 0

    for desc in descriptions:
        lines = str(desc).split("\n")
        empty_lines += sum(1 for line in lines if not line.strip())
        caps = split_captions(desc)
        per_image_counts[len(caps)] += 1
        duplicate_within_image += len(caps) - len(set(c.lower() for c in caps))
        for cap in caps:
            if any(ord(ch) > 127 for ch in cap):
                non_ascii_captions += 1
            raw_chars.update(ch for ch in cap if not ch.isalnum() and not ch.isspace())
            tokens = normalize(cap).split()
            lengths.append(len(tokens))
            word_counts.update(tokens)
            bigram_counts.update(zip(tokens, tokens[1:]))

    import numpy as np
    arr = np.array(lengths)
    content = Counter({w: c for w, c in word_counts.items() if w not in STOPWORDS})
    return {
        "num_captions": len(lengths),
        "captions_per_image": dict(sorted(per_image_counts.items())),
        "empty_lines": empty_lines,
        "duplicate_captions_within_image": duplicate_within_image,
        "non_ascii_captions": non_ascii_captions,
        "length_words": {
            "min": int(arr.min()), "max": int(arr.max()), "mean": round(float(arr.mean()), 2),
            "median": float(np.median(arr)), "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        },
        "vocab_size": len(word_counts),
        "words_appearing_once": sum(1 for c in word_counts.values() if c == 1),
        "words_appearing_lt5": sum(1 for c in word_counts.values() if c < 5),
        "top_words": content.most_common(top_k),
        "color_word_counts": {c: word_counts.get(c, 0) for c in COLOR_WORDS},
        "punctuation_counts": dict(raw_chars.most_common(15)),
        "_lengths": lengths,
        "_word_counts": word_counts,
        "_bigram_counts": bigram_counts,
    }


def clean_all(ids: list, descriptions: list[str], spell_fixes: dict[str, str]) -> dict[str, list[str]]:
    """Map each image id to its cleaned captions."""
    return {str(i): [clean_caption(c, spell_fixes) for c in split_captions(d)]
            for i, d in zip(ids, descriptions)}
