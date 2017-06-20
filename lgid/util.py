
from collections import defaultdict
import unicodedata
import re

def read_language_table(path):
    table = defaultdict(list)
    for line in open(path):
        if line.strip():
            name, codes = line.split('\t', 1)
            codes = codes.split()
            name = normalize_characters(name)         # remove diacritics
            name = re.sub(r' \([^)]*\)', '', name)    # remove parentheticals
            name = re.sub(r'\s+', ' ', name).strip()  # normalize spacing
            table[name].extend(codes)
    for name in table:
        table[name] = sorted(set(table[name]))  # remove duplicates
    return table

def normalize_characters(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s)
                     if not unicodedata.combining(c))
