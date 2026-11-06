"""Text statistics for the example toolpack.

Pure functions over a string: no I/O, no state, nothing outside the
toolpack import allowlist (re, collections, math).
"""
import math
import re
from collections import Counter

WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
SENTENCE_PATTERN = re.compile(r"[.!?]+")
WORDS_PER_MINUTE = 200.0


def find_words(text):
    return WORD_PATTERN.findall(text)


def text_stats(text):
    """Counts and a reading-time estimate for ``text``."""
    words = find_words(text)
    word_count = len(words)
    unique = {w.lower() for w in words}
    total_length = sum(len(w) for w in words)
    sentences = len(SENTENCE_PATTERN.findall(text))
    if word_count and not sentences:
        sentences = 1
    paragraphs = len([p for p in text.split("\n\n") if p.strip()])
    minutes = word_count / WORDS_PER_MINUTE
    return {
        "characters": len(text),
        "characters_no_spaces": len(text.replace(" ", "")),
        "words": word_count,
        "unique_words": len(unique),
        "sentences": sentences,
        "paragraphs": paragraphs,
        "average_word_length": (total_length / word_count) if word_count else 0.0,
        "reading_time_minutes": round(minutes, 3),
        "reading_time_seconds": math.ceil(minutes * 60),
    }


def word_frequency(text, top_n=5):
    """The ``top_n`` most frequent words, as a word -> count mapping
    ordered by count (descending), then word (ascending)."""
    counts = Counter(w.lower() for w in find_words(text))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:top_n])
