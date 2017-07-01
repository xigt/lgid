
"""
Utility functions for language idenfication tasks
"""

from collections import defaultdict
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
    for line in open(path):
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

def read_odin_language_model(path):
    """
    Read an ODIN language model (built from training data) at *path*
    """
    pass

def read_crubadan_language_model(path):
    """
    Read a Crubadan language model at *path*
    """
    pass
