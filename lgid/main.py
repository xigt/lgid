#!/usr/bin/env python3

USAGE = '''
Usage: lgid [-v...] [--mentions] CONFIG INFILE...

Arguments:
  CONFIG                    path to a config file
  INFILE                    a Freki-formatted text file

Options:
  -h, --help                print this usage and exit
  -v, --verbose             increase logging verbosity
  --mentions                just print language mentions that were found

Examples:

  lgid -v parameters.conf 123.freki

'''

from configparser import ConfigParser
import logging

import docopt

from freki.serialize import FrekiDoc

from lgid.util import (
    read_language_table,
    read_odin_language_model,
    read_crubadan_language_model
)
from lgid.analyzers import (
    language_mentions,
    character_ngrams,
    word_ngrams
)
from lgid.features import (
    gl_features,
    w_features,
    l_features,
    m_features,
)


def main():
    args = docopt.docopt(USAGE)
    logging.basicConfig(level=50 - ((args['--verbose'] + 2) * 10))

    config = ConfigParser()
    config.read(args['CONFIG'])

    locs = config['locations']

    lgtable = {}
    if locs['language-table']:
        lgtable = read_language_table(locs['language-table'])

    olm = None
    if locs['odin-language-model'] and not args['--mentions']:
        olm = read_odin_language_model(locs['odin-language-model'])

    clm = None
    if locs['crubadan-language-model'] and not args['--mentions']:
        clm = read_crubadan_language_model(locs['crubadan-language-model'])

    for infile in args['INFILE']:
        
        doc = FrekiDoc.read(infile)

        context = {}
        context['last-lineno'] = max(x.lineno for x in doc.lines())

        caps = config['parameters'].get('mention-capitalization', 'default')
        lgmentions = list(language_mentions(doc, lgtable, caps))
        if args['--mentions']:
            for m in lgmentions:
                print('\t'.join(map(str, m)))
            return

        features_template = dict(((m.name, m.code), {}) for m in lgmentions)
        for span in spans(doc):
            if not span:
                continue
            context['span-top'] = span[0].lineno
            context['span-bottom'] = span[-1].lineno
            for line in span:
                features = dict(features_template)
                context['line'] = line
                # print(character_ngrams(line, 3))
                # print(word_ngrams(line, 2))
                if '-L' in line.tag:
                    gl_features(features, lgmentions, context, config)
                    w_features(features, lgmentions, context, config)
                    l_features(features, lgmentions, line, config)
                if '-G' in line.tag:
                    pass
                if '-M' in line.tag:
                    m_features(features, lgmentions, line, config)

                    # l_features(features, olm, clm, context, config)
                print(line.lineno)
                print(features)
                    # w_before_mentions(features, lgmentions, line.lineno, params)
                    # w_after_mentions(features, lgmentions, line.lineno, params)

def spans(doc):
    span = []
    for line in doc.lines():
        if line.tag.startswith('B-'):
            if span:
                yield span
                span = []
            span.append(line)
        elif line.tag.startswith('I-'):
            span.append(line)
        elif span:
            yield span
            span = []
    if span:
        yield span


if __name__ == '__main__':
    main()
