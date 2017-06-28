
from collections import defaultdict
import unicodedata
import re
import logging

def read_language_table(path):
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
    return ''.join(c for c in unicodedata.normalize('NFKD', s)
                     if not unicodedata.combining(c))

def read_odin_language_model(path):
    pass

def read_crubadan_language_model(path):
    pass
