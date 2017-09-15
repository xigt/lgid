
"""
Utility functions for language idenfication tasks
"""

from collections import defaultdict
from freki.serialize import FrekiDoc
import unicodedata
import re
import logging

def read_language_table(path):
    """
    Read language table at *path* and return the {name:[code]} mapping

    Language names are normalized to remove diacritics, parentheticals,
    and excess spacing. The result is lowercased to avoid hash-misses
    based on capitalization differences.

    Args:
        path: file path to a tab-separated language table where the
            first column is the language name and following columns
            contain language codes for that name
    Returns:
        dictionary mapping normalized language names to lists of codes
    """
    logging.info('Reading language table: ' + path)

    table = defaultdict(list)
    for line in open(path, encoding='utf-8'):
        if line.strip():
            name, codes = line.rstrip().split('\t', 1)
            codes = codes.split()
            norm = normalize_characters(name)         # remove diacritics
            norm = re.sub(r' \([^)]*\)', '', norm)    # remove parentheticals
            norm = re.sub(r'\s+', ' ', norm).strip()  # normalize spacing
            norm = norm.lower()                       # lowercase
            table[norm].extend(codes)
    for norm in table:
        table[norm] = sorted(set(table[norm]))  # remove duplicates
    logging.info(str(len(table)) + ' language names in table')
    return table

def normalize_characters(s):
    """
    Apply a unicode transformation to normalize accented characters to
    their near-ASCII equivalent.
    """
    return ''.join(c for c in unicodedata.normalize('NFKD', s)
                     if not unicodedata.combining(c))

def read_odin_language_model(pairs, config, characters):
    """
    Read an ODIN language model for a (language name, ISO code) pair.

    Args:
        lang_name: unicode-normalized language name
        iso_code: an ISO code
        config: model parameters
        characters: whether to use character or word ngrams
    Returns:
        list of tuples of ngrams, or None if no language model exists for
        the given name-ISO pairing
    """
    all_lms = {}
    for lang_name, iso_code in pairs:
        lang_name = lang_name.replace('/', '-')
        base_path = config['locations']['odin-language-model']

        if characters:
            file_name = '{}/{}_{}.char'.format(base_path, iso_code, lang_name)
            n = int(config['parameters']['character-n-gram-size'])
        else:
            file_name = '{}/{}_{}.word'.format(base_path, iso_code, lang_name)
            n = int(config['parameters']['word-n-gram-size'])
        file_name = file_name.encode('ascii', 'ignore').decode('ascii')
        try:
            with open(file_name, encoding='utf8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            continue

        lm = set()
        for line in lines:
            if line.strip() == '':
                continue
            line = line.split()[0] if characters else line.split()[:-1]
            if len(line) <= n:
                feature = tuple(line)
                lm.add(feature)
        all_lms[(lang_name, iso_code)] = lm
    return all_lms

def read_crubadan_language_model(pairs, config, characters):
    """
    Read a Crubadan language model for a (language name, ISO code) pair.

    Args:
        lang_name: unicode-normalized language name
        iso_code: an ISO code
        config: model parameters
        characters: whether to use character or word ngrams
    Returns:
        list of tuples of ngrams, or None if no language model exists for
        the given name-ISO pairing
    """
    import csv

    base_path = config['locations']['crubadan-language-model']

    table = open(config['locations']['crubadan-directory-index'], encoding='utf8')
    reader = csv.reader(table)
    header = next(reader) # discard header row

    dir_map = {}
    for row in reader:
        name = row[0]
        iso = row[1]
        directory = row[2].strip()
        dir_map[(name, iso)] = directory
    table.close()

    if characters:
        file_basename = {
            3: "-chartrigrams.txt"
        }.get(int(config['parameters']['crubadan-char-size']))
    else:
        file_basename = {
            1: "-words.txt",
            2: "-wordbigrams.txt"
        }.get(int(config['parameters']['crubadan-word-size']))

    all_lms = {}
    for lang_name, iso_code in pairs:
        try:
            this_dir = dir_map[(lang_name, iso_code)]
            crubadan_code = this_dir.split("_")[1]
            with open("{}/{}/{}{}".format(base_path, this_dir, crubadan_code, file_basename), encoding='utf8') as f:
                lines = f.readlines()
        except (FileNotFoundError, KeyError, IndexError):
            continue

        lm = set()
        for line in lines:
            if line.strip() == '':
                continue
            line = line.split()[:-1]
            feature = tuple(line[0]) if characters else tuple(line)
            lm.add(feature)
        all_lms[(lang_name, iso_code)] = lm
    return all_lms

def encode_instance_id(doc_id, span_id, line_no, lang_name, lang_code):
    return (doc_id, span_id, line_no, lang_name, lang_code)

def decode_instance_id(s):
    doc_id, span_id, line_no, lang_name, lang_code = s.id
    return doc_id, span_id, int(line_no), lang_name, lang_code


def spans(doc):
    """
    Scan the FrekiDoc *doc* and yield the IGT spans found

    This requires the documents to have the span_id attribute from
    IGT detection.
    """
    span = []
    span_id = None
    for line in doc.lines():
        new_span_id = line.attrs.get('span_id')
        if new_span_id != span_id and span:
            yield span
            span = []
            span_id = new_span_id
        if new_span_id is not None:
            span.append(line)
            span_id = new_span_id
    if span:
        yield span


def find_common_codes(infiles, config):
    """
    Build the res file showing the most common code for each language
    :param infiles: list of freki filepaths
    :param config: config object
    :return: None, writes to most-common-codes location
    """
    dialect_count = defaultdict(lambda: defaultdict(int))
    locs = config['locations']
    lgtable = {}
    if locs['language-table']:
        lgtable = read_language_table(locs['language-table'])
    for infile in infiles:
        doc = FrekiDoc.read(infile)
        for span in spans(doc):
            if not span:
                continue
            for line in span:
                if 'L' in line.tag:
                    lgname = line.attrs.get('lang_name', '???').lower()
                    lgcode = line.attrs.get('lang_code', 'und')
                    if len(lgcode.split(':')) > 1:
                        parts = lgcode.split(':')
                        for part in parts:
                            dialect_count[lgname][part] += 1
                    else:
                        dialect_count[lgname][lgcode] += 1
    out = open(locs['most-common-codes'], 'w')
    for key in lgtable:
        a_line = ''
        if len(lgtable[key]) == 1:
            a_line = key + '\t'
            a_line += lgtable[key][0]
            a_line += '\n'
        if dialect_count[key]:
            a_line = key + '\t'
            a_line += sorted(dialect_count[key], key=lambda x: dialect_count[key][x], reverse=True)[0]
            a_line += '\n'
        out.write(a_line)

