"""
-----------------------------------------------------------------------
LINKGRAM PARSING UTILS
-----------------------------------------------------------------------

TODO: LG Splits up on punctuation
"""

from functools import lru_cache
import re

import linkgrammar as lg
from linkgrammar import Sentence

# this regex splits larger inputs into sentence sized spans before they are passed into link grammar.
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+(?=(?:["\'])?[A-Z0-9])|\n+')

#memoize loading of the link grammar dictionary and parse options
@lru_cache(maxsize=1)
def _get_linkgrammar_runtime() -> tuple[lg.Dictionary, lg.ParseOptions]:
    return lg.Dictionary("en"), lg.ParseOptions(verbosity=0, linkage_limit=1, max_parse_time=10)

"""
Removes leading and trailing whitespace from the span boundaries.
"""
def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while start < end and text[end - 1].isspace():
        end -= 1
    return start, end

"""
Splits a larger text into sentence sized spans before it is passed into link grammar.

This keeps the parser working on smaller chunks and also prevents links from
being built across sentence boundaries.
"""
def split_sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    sentence_start = 0

    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        start, end = _trim_span(text, sentence_start, match.start())
        if start < end:
            spans.append((start, end))
        sentence_start = match.end()

    start, end = _trim_span(text, sentence_start, len(text))
    if start < end:
        spans.append((start, end))

    return spans


"""
If link grammar fails on a sentence, fall back to simple whitespace spans.
"""
def _fallback_word_spans(text: str, text_start: int) -> list[tuple[int, int]]:
    return [
        (text_start + match.start(), text_start + match.end())
        for match in re.finditer(r"\S+", text)
    ]

"""
Parses one sentence sized span and converts it into the two things we need:
* the word spans in the original text
* the links between those words
"""
def parse_sentence_features(
    sentence_text: str,
    sentence_start: int,
    link_type_to_id: dict[tuple[str, str], int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    dictionary, parse_options = _get_linkgrammar_runtime()

    # if parsing fails, we still return word spans so later alignment code can continue.
    try:
        linkages = Sentence(sentence_text, dictionary, parse_options).parse()
        linkage = next(iter(linkages), None)
    except Exception:
        linkage = None

    if linkage is None:
        return _fallback_word_spans(sentence_text, sentence_start), []

    word_spans: list[tuple[int, int]] = []
    local_to_compact: dict[int, int] = {}

    # gather the real words only and ignore parser specific wall tokens.
    for word_index in range(linkage.num_of_words()):
        word_text = linkage.word(word_index)
        word_start = linkage.word_char_start(word_index)
        word_end = linkage.word_char_end(word_index)
        if word_text in {"LEFT-WALL", "RIGHT-WALL"} or word_start >= word_end:
            continue

        local_to_compact[word_index] = len(word_spans)
        word_spans.append((sentence_start + word_start, sentence_start + word_end))

    links: list[tuple[int, int, int]] = []

    # convert each parser link into the compact word indices we built above.
    for link in linkage.links():
        left_word_index = lg.Clinkgrammar.linkage_get_link_lword(linkage._obj, link.index)
        right_word_index = lg.Clinkgrammar.linkage_get_link_rword(linkage._obj, link.index)
        if left_word_index not in local_to_compact or right_word_index not in local_to_compact:
            continue

        link_key = (link.left_label, link.right_label)
        # append to global link type dictionary if it's not already there.
        link_type_id = link_type_to_id.setdefault(link_key, len(link_type_to_id))
        links.append(
            (
                local_to_compact[left_word_index],
                local_to_compact[right_word_index],
                link_type_id,
            )
        )

    return word_spans, links
