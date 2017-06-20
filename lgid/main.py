#!/usr/bin/env python3

USAGE = '''
Usage: lgid [-v...] [--lang-table=PATH] INFILE...

Arguments:
  INFILE                    a Freki-formatted text file

Options:
  -h, --help                print this usage and exit
  -v, --verbose             increase logging verbosity
  --lang-table PATH         path to a tab-separated language table

'''

import logging

import docopt

from freki.serialize import FrekiDoc

from lgid.util import read_language_table
from lgid.analyzers import (
    language_mentions,
    character_ngrams,
    word_ngrams
)
# from lgid.features import language_mentions


def main():
    args = docopt.docopt(USAGE)
    logging.basicConfig(level=50 - ((args['--verbose'] + 2) * 10))

    if args['--lang-table']:
        logging.debug('Reading language table: ' + args['--lang-table'])
        lgtable = read_language_table(args['--lang-table'])

    for infile in args['INFILE']:
        
        doc = FrekiDoc.read(infile)

        logging.debug('Finding language mentions')
        lgmentions = language_mentions(doc, lgtable)

        for mention in lgmentions:
            print(mention)
            
        # print(list(lgmentions))

        for block in doc.blocks:
            for line in block.lines:
                pass
                # print(character_ngrams(line, 3))
                # print(word_ngrams(line, 2))
                # if '-L' in line.tag:
                #     features = []


if __name__ == '__main__':
    main()
