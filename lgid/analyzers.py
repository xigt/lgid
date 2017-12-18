
"""
Document analyzers to assist feature functions

Functions in this module extract information from documents so that
feature functions can use the information to determine which features
should be turned on.
"""

import re
from collections import namedtuple
import logging
from string import punctuation

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


def language_mentions(doc, lgtable, lang_mapping_tables, capitalization, single_mention):
    """
    Find mentions of languages in a document

    When a matching language name is detected in a document, information
    about that mention is recorded, such as the position, the normalized
    name and language code, and the original string that matched.

    Args:
        doc: FrekiDoc instance
        lgtable: mapping of normalized language names to a list of
            language codes (e.g., from lgid.util.read_language_table())
        lang_mapping_tables: tuple of tables mapping language names to ints
        capitalization: scheme for normalizing language names; valid
            values are 'upper', 'lower', and 'title'. Using 'title'
            (uppercase the first letter of each token) helps prevent
            word-like language names (Even, She, Day, etc.) from
            over-firing.
        single_mention: if True, finds all mentions; if False, finds only the
            longest mention within each string
    Yields:
        Mention objects
    """
    logging.info('Finding language mentions')

    normcaps = {
        'upper': str.upper,
        'lower': str.lower,
        'title': str.title
    }.get(capitalization, str)

    k = 0
    for block in doc.blocks:
        logging.debug(block.block_id)
        for i, line1 in enumerate(block.lines):
            added_space = 0
            # combine two lines so we can do multiline matching
            if i + 1 < len(block.lines):
                line2 = block.lines[i + 1]
                endline = line2.lineno

                if line1.endswith('-') or line2.startswith('-'):
                    lines = line1.rstrip(' ') + line2.lstrip(' ')
                else:
                    lines = line1.rstrip(' -') + ' ' + line2.lstrip(' -')
                    added_space = 1
            else: # line 1 is the last line in the block
                line2 = None
                endline = line1.lineno
                lines = line1.rstrip(' -')

            # map each word in the line to either its word id or 'N' if it's not
            # in the vocabulary
            mapped_lines = ''
            for w in lines.split():
                w = normcaps(normalize_characters(w))
                w = re.sub(r"^.*?(\w+(-\w+)?).*$", "\g<1>", w)
                if w in lang_mapping_tables.word_to_int:
                    mapped_lines += lang_mapping_tables.word_to_int[w]
                elif '-' in w:
                    w = normcaps(w.replace('-', ''))
                    if w in lang_mapping_tables.word_to_int:
                        mapped_lines += lang_mapping_tables.word_to_int[w]
                    else:
                        mapped_lines += 'N'
                else:
                    mapped_lines += 'N'
            language_strings = [x for x in mapped_lines.split('N') if x != '']

            # get the index in the string of each word that's in the vocabulary
            result_locs = []
            pos, j = 0, 0
            while j < len(mapped_lines):
                if mapped_lines[j] == 'N':
                    j += 1
                else:
                    result_locs.append(pos)
                    j += 5
                pos += 1

            # add together words to find all the language matches
            total_matched = [] # list of every match on this line
            language_spans = [] # list of tuples of the span of each language, referencing indices from result_locs
            last_span = (0, 0)
            for result in language_strings:
                matched = []
                current_span = [last_span[1], last_span[1]]
                j = 0
                language = ''
                while j < len(result):
                    num = result[j:j + 5]
                    word = lang_mapping_tables.int_to_word[num]
                    language += word + ' '
                    j += 5
                    current_span[1] += 1
                    if language.strip() in lang_mapping_tables.lang_to_int:
                        matched.append(language.strip())
                        language_spans.append(tuple(current_span))
                full_language = language.split()
                idx = 1
                for lg in full_language[1:-1]:
                    if lg.strip() in lang_mapping_tables.lang_to_int:
                        matched.append(lg.strip())
                        language_spans.append((current_span[0] + idx, current_span[0] + idx + 1))
                    idx += 1
                while j > 5:
                    language = ' '.join(language.split()[1:])
                    j -= 5
                    current_span[0] += 1
                    if language.strip() in lang_mapping_tables.lang_to_int:
                        matched.append(language.strip())
                        language_spans.append(tuple(current_span))
                if matched == []:
                    last_span = tuple(current_span)
                    continue
                elif not single_mention:
                    last_span = tuple(current_span)
                    total_matched += matched
                else:
                    matched.sort(key=lambda x: len(x.split()), reverse=True)
                    last_span = tuple(current_span)
                    total_matched.append(matched[0])

            # calculate the character spans of each in-vocab word on the line, referencing indices from result_locs
            word_idx = 0
            try:
                char_number = lines.index(lines.lstrip()[0])
            except IndexError:
                char_number = 0
            char_locs = []
            last_char = ''
            w_start, w_end = -1, -1
            added = False
            new_word = True
            for char in list(lines.lstrip()):
                if last_char == ' ' and char != ' ':
                    word_idx += 1
                    added = False
                    new_word = True
                if word_idx in result_locs and new_word:
                    w_start = char_number
                    if char_number + 1 == len(lines) or (char == '+' and last_char == ' '):
                        char_locs.append((w_start, w_start + 1))
                elif not added and word_idx in result_locs and \
                        ((char == ' ' or (char in punctuation and char != '-'))
                           and
                           (last_char != ' ' and (last_char not in punctuation or last_char == '-'))):
                    w_end = char_number
                    char_locs.append((w_start, w_end))
                    added = True
                elif not added and word_idx in result_locs and char_number + 1 == len(lines):
                    w_end = char_number + 1
                    char_locs.append((w_start, w_end))
                    added = True
                char_number += 1
                new_word = False
                last_char = char

            # for each mention, form a tuple of (name, character span)
            # character span is itself a tuple of column numbers
            annotated_matches = []
            for j, match in enumerate(total_matched):
                char_span = (char_locs[language_spans[j][0]][0], char_locs[language_spans[j][1] - 1][1])
                annotation = (match, char_span)
                annotated_matches.append(annotation)

            # iterate through each match and create a Mention object out of it
            startline = line1.lineno
            line_break = len(line1.rstrip(' -'))
            for match in annotated_matches:
                k += 1
                name = match[0].lower()
                start, end = match[1][0], match[1][1]

                orig_end = end
                this_startline, this_endline = startline, endline
                if start < line_break and end > line_break and line1 and line2: # match crosses lines
                    orig_end -= len(line1) + added_space
                    orig_end += len(line2) - len(line2.lstrip(' -')) # account for leading whitespace on Freki lines
                    text = line1[start:] + line2[:orig_end]
                elif end <= line_break: # match is only in line 1
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
    logging.info(str(k) + ' language mentions found')


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
                ngrams.append(tuple(word[i:i+j]))

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
        ngrams.append(tuple(words[i:i+n]))

    return ngrams

def morpheme_ngrams(s, n, splitter, lhs='', rhs=''):
    """
    Extract morpheme n-grams of length *n* from string *s*

    Args:
        s: the string whence n-grams are extracted
        n: the length of each n-gram
        lhs: left-padding character (to show token boundaries)
        rhs: right-padding character (to show token boundaries)
    Returns:
        list of n-grams in *s*
    """
    ngrams = []

    words = [lhs] + re.split(splitter, s) + [rhs]

    rangemax = len(words) - (n-1)
    if rangemax < 1:
        rangemax = 1

    for i in range(rangemax):
        ngrams.append(tuple(words[i:i+n]))

    return ngrams
