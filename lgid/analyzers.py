
import re

from lgid.util import normalize_characters


def language_mentions(doc, lgtable):
    lg_re = re.compile(
        r'\b({})\b'.format(r'|'.join(re.escape(name) for name in lgtable)),
        flags=re.U|re.I
    )
    for block in doc.blocks:
        # page = block.page
        for line in block.lines:
            match = lg_re.search(normalize_characters(line))
            if match is not None:
                name = match.group(0)
                start, end = match.span()
                yield ((line.lineno, start), (line.lineno, end), name, lgtable[name.lower()])


def character_ngrams(s, n, lhs='<', rhs='>'):
    ngrams = []

    for word in s.split():
        word = lhs + word + rhs

        rangemax = len(word) - (n-1)
        if rangemax < 1:
            rangemax = 1

        for i in range(rangemax):
            ngrams.append(word[i:i+n])

    return ngrams


def word_ngrams(s, n, lhs='\\n', rhs='\\n'):
    ngrams = [] 

    words = [lhs] + s.split() + [rhs]

    rangemax = len(words) - (n-1)
    if rangemax < 1:
        rangemax = 1

    for i in range(rangemax):
        ngrams.append(words[i:i+n])

    return ngrams
