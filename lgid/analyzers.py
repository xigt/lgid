
"""
Document analyzers to assist feature functions

Functions in this module extract information from documents so that
feature functions can use the information to determine which features
should be turned on.
"""

import re
from collections import namedtuple
import logging
import pickle

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


def find_language_mentions(doc, lgtable, lang_mapping_tables, capitalization):
    """
    Find mentions of languages in a document

    When a matching language name is detected in a document, information
    about that mention is recorded, such as the position, the normalized
    name and language code, and the original string that matched.

    Args:
        doc: FrekiDoc instance
        lgtable: mapping of normalized language names to a list of
            language codes (e.g., from lgid.util.read_language_table())
        capitalization: scheme for normalizing language names; valid
            values are 'upper', 'lower', and 'title'. Using 'title'
            (uppercase the first letter of each token) helps prevent
            word-like language names (Even, She, Day, etc.) from
            over-firing.
    Yields:
        Mention objects
    """
    logging.info('Finding language mentions')

    normcaps = {
        'upper': str.upper,
        'lower': str.lower,
        'title': str.title
    }.get(capitalization, str)

    space_hyphen_re = re.compile(r'[-\s]+', flags=re.U)
    lg_re = re.compile(
        r'\b({})\b'.format(
            r'|'.join(
                r'[-\s]*'.join(
                    map(re.escape, map(normcaps, re.split(space_hyphen_re, name))
                )
            ) for name in lgtable)
        ),
        flags=re.U
     )
    i = 0
    for block in doc.blocks:
        logging.debug(block.block_id)
        for i, line1 in enumerate(block.lines):
            if i + 1 < len(block.lines):
                line2 = block.lines[i + 1]
                endline = line2.lineno
                lines = line1.rstrip(' -') + line2.lstrip(' -')
            else: # line 1 is the last line in the block
                line2 = None
                endline = line1.lineno
                lines = line1.rstrip(' -')

            # speedy
            int_to_lang = lang_mapping_tables[1]
            lang_to_int = lang_mapping_tables[0]
            idlines = ''
            for w in lines.split():
                w = normcaps(w)
                if w in lang_to_int:
                    idlines += lang_to_int[w] # issue
                else:
                    idlines += 'N'
            idlines = [x for x in idlines.split('N') if x != '']

            for result in idlines:
                i = 0
                language = ''
                while i < len(result):
                    num = result[i:i + 5]
                    word = int_to_lang[num]
                    language += word + ' '
                    i += 5
                print(language)

            #end speedy

            startline = line1.lineno
            line_break = len(line1.rstrip(' -'))
            for match in re.finditer(lg_re, normalize_characters(lines)):
                i += 1
                name = match.group(0).lower()
                start, end = match.span()

                space_hyphen_count = 0
                end_loop = end
                for j, char in enumerate(lines[start:], start=start):
                    if j == end_loop - 1:
                        break
                    if char == '-' or char == ' ':
                        space_hyphen_count += 1
                        end_loop += 1

                orig_end = end + space_hyphen_count
                this_startline, this_endline = startline, endline
                hyphen_name = None
                if start < line_break and end > line_break and line1 and line2: # match crosses lines
                    orig_end -= len(line1) - 1
                    orig_end += len(line2) - len(line2.lstrip(' -')) # account for leading whitespace on Freki lines
                    text = line1[start:] + line2[:orig_end]
                    hyphen_name = name[:line_break - start] + "-" + name[line_break - start:]
                    space_name = name[:line_break - start] + " " + name[line_break - start:]
                elif end < line_break: # match is only in line 1
                    this_endline = line1.lineno
                    text = line1[start:orig_end]
                else: # match is only in line 2
                    continue # if we include matches only in line2, they'll be doubled

                lg_codes = lgtable[name]
                if lg_codes != []:
                    for code in lg_codes:
                        yield Mention(
                            this_startline, start, this_endline, orig_end, name, code, text
                        )
                elif hyphen_name:
                    lg_codes = lgtable[hyphen_name]
                    if lg_codes != []:
                        for code in lg_codes:
                            yield Mention(
                                this_startline, start, this_endline, orig_end, hyphen_name, code, text
                            )
                    else:
                        for code in lgtable[space_name]:
                            yield Mention(
                                this_startline, start, this_endline, orig_end, space_name, code, text
                            )
    logging.info(str(i) + ' language mentions found')


def language_mentions(doc, lgtable, capitalization):
    key = str(doc)[:20]
    try:
        mention_dict = pickle.load(open('mentions.p', 'rb'))
    except FileNotFoundError:
        mention_dict = {}
    if key not in mention_dict:
        mentions = list(find_language_mentions(doc, lgtable, capitalization))
        mention_dict[key] = mentions
        pickle.dump(mention_dict, open('mentions.p', 'wb'))
    return mention_dict[key]


def character_ngrams(s, ngram_range, lhs='<', rhs='>'):
    """
    Extract character n-grams of length *n* from string *s*

    Args:
        s: the string whence n-grams are extracted
        ngram_range: tuple with two elements: the min and max ngram lengths
        lhs: left-padding character (to show token boundaries)
        rhs: right-padding character (to show token boundaries)
    Returns:
        list of n-grams in *s*
    """
    ngrams = []

    for word in s.split():
        word = lhs + word + rhs

        rangemax = len(word) - (ngram_range[1]-1)
        if rangemax < 1:
            rangemax = 1

        for i in range(rangemax):
            for j in range(ngram_range[0], ngram_range[1] + 1):
                ngrams.append(word[i:i+j])

    return ngrams


def word_ngrams(s, n, lhs='\\n', rhs='\\n'):
    """
    Extract word n-grams of length *n* from string *s*

    Args:
        s: the string whence n-grams are extracted
        n: the length of each n-gram
        lhs: left-padding character (to show token boundaries)
        rhs: right-padding character (to show token boundaries)
    Returns:
        list of n-grams in *s*
    """
    ngrams = [] 

    words = [lhs] + s.split() + [rhs]

    rangemax = len(words) - (n-1)
    if rangemax < 1:
        rangemax = 1

    for i in range(rangemax):
        ngrams.append(words[i:i+n])

    return ngrams
