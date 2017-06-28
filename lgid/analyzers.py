
import re
from collections import namedtuple
import logging

from lgid.util import normalize_characters

Mention = namedtuple(
    'Mention',
    ('startline',   # line number of the start of the match
     'startcol',    # column of the start of the match
     'endline',     # line number of the end of the match
     'endcol',      # column of the end of the match
     'name',        # normalized language name
     'code',        # language code
     'text')        # original text of matched name
)

def language_mentions(doc, lgtable, capitalization):
    logging.info('Finding language mentions')

    normcaps = {
        'upper': str.upper,
        'lower': str.lower,
        'title': str.title
    }.get(capitalization, str)

    lg_re = re.compile(
        r'\b({})\b'.format(
            r'|'.join(re.escape(normcaps(name)) for name in lgtable)
        ),
        flags=re.U
    )
    i = 0
    for block in doc.blocks:
        logging.debug(block.block_id)
        # page = block.page
        for line in block.lines:
            startline = line.lineno
            endline = line.lineno  # same for now
            match = lg_re.search(normalize_characters(line))
            if match is not None:
                i += 1
                name = match.group(0).lower()
                start, end = match.span()
                text = line[start:end]
                for code in lgtable[name]:
                    yield Mention(
                        startline, start, endline, end, name, code, text
                    )
    logging.info(str(i) + ' language mentions found')


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
