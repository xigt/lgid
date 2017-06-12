#!/usr/bin/env python3

USAGE = '''
Usage: lgid INFILE...

Arguments:
  INFILE                    a Freki-formatted text file

Options:
  -h, --help                print this usage and exit
  --lang-table PATH         path to a tab-separated language table

'''

import docopt

from lgid.features import language_mention


def main():
    args = docopt.docopt(USAGE)

    for infile in args['INFILE']:
        
        for line in open(infile):
            mention.analyze()


if __name__ == '__main__':
    main()
