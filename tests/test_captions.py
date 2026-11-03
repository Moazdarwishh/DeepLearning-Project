from collections import Counter

import pytest

from src.data.captions import caption_statistics, clean_all, clean_caption, normalize


def test_normalize_strips_punctuation_digits_and_accents():
    assert normalize("This Bird, has 2 WINGS!! café") == "this bird has wings cafe"


def test_clean_caption_applies_fixes():
    assert clean_caption("a blakc bird", {"blakc": "black"}) == "a black bird"


def test_caption_statistics_counts():
    stats = caption_statistics(["a red bird.\nA red bird.\n\nsmall blue bird"])
    assert stats["captions_per_image"] == {3: 1}
    assert stats["duplicate_captions_within_image"] == 1
    assert stats["empty_lines"] == 1
    assert stats["length_words"]["max"] == 3


def test_clean_all_keeps_every_caption():
    out = clean_all([7], ["One.\nTwo!"], {})
    assert out == {"7": ["one", "two"]}


def test_spell_fixes_protect_domain_words():
    pytest.importorskip("spellchecker")
    from src.data.captions import build_spell_fixes
    wc = Counter({"black": 500, "blakc": 1, "superciliary": 1, "malar": 2})
    fixes = build_spell_fixes(wc)
    assert fixes.get("blakc") == "black"
    assert "superciliary" not in fixes and "malar" not in fixes


def test_merged_words_and_apostrophes():
    pytest.importorskip("spellchecker")
    from src.data.captions import build_spell_fixes
    wc = Counter({"a": 900, "gray": 300, "red": 400, "with": 800, "also": 500, "agray": 1, "redwith": 1,
                  "aalso": 1, "bicolored": 1})
    bigrams = Counter({("a", "gray"): 200, ("red", "with"): 90})
    fixes = build_spell_fixes(wc, bigrams)
    assert fixes["agray"] == "a gray" and fixes["redwith"] == "red with"
    assert fixes.get("aalso") == "also"
    assert "bicolored" not in fixes
    assert clean_caption("this bird�s wings agray", fixes) == "this birds wings a gray"
