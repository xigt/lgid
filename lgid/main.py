#!/usr/bin/env python3

USAGE = '''
Usage:
  lgid [-v...] train    --model=PATH CONFIG INFILE...
  lgid [-v...] classify --model=PATH CONFIG INFILE...
  lgid [-v...] test     --model=PATH CONFIG INFILE...
  lgid [-v...] list-mentions         CONFIG INFILE...

Commands:
  train                     train a model from supervised data
  test                      test on new data using a saved model
  classify                  output predictions on new data using a saved model
  list-mentions             just print language mentions from input files

Arguments:
  CONFIG                    path to a config file
  INFILE                    a Freki-formatted text file

Options:
  -h, --help                print this usage and exit
  -v, --verbose             increase logging verbosity
  --model PATH              where to save/load a trained model

Examples:

  lgid -v train --model=model.gz parameters.conf 123.freki 456.freki
  lgid -v test --model=model.gz parameters.conf 789.freki
  lgid -v classify --model=model.gz parameters.conf 1000.freki
  lgid -v list-mentions parameters.conf 123.freki

'''

import os
from configparser import ConfigParser
import logging

import docopt

from freki.serialize import FrekiDoc

from lgid.models import (
    StringInstance,
    LogisticRegressionWrapper as Model,
    chi2
)

from lgid.util import (
    read_language_table,
    read_odin_language_model,
    read_crubadan_language_model,
    encode_instance_id,
    decode_instance_id
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

    modelpath = args['--model']
    infiles = args['INFILE']

    if args['train']:
        train(infiles, modelpath, config)
    elif args['test']:
        test(infiles, modelpath, config)
    elif args['list-mentions']:
        list_mentions(infiles, config)


def train(infiles, modelpath, config):
    """
    Train a language-identification model from training data

    Args:
        infiles: iterable of Freki file paths
        modelpath: the path where the model will be written
        config: model parameters
    """
    instances = list(get_instances(infiles, config))
    # for inst in instances:
    #     print(inst.id, inst.label)
    model = Model()
    model.feat_selector = chi2
    model.train(instances)
    model.save(modelpath)
    # for dist in model.test(instances):
    #     print(dist.best_class, dist.best_prob, len(dist.dict))

def test(infiles, modelpath, config):
    """
    Test a language-identification model

    Args:
        infiles: iterable of Freki file paths
        modelpath: the path where the model will be loaded
        config: model parameters
    """
    instances = list(get_instances(infiles, config))
    model = Model()
    model.load(modelpath)
    for dist in model.test(instances):
        print(dir(dist))
        print(dist.classes())

def list_mentions(infiles, config):
    """
    List all language mentions found in the given files

    Args:
        infiles: iterable of Freki file paths
        config: model parameters
    """
    lgtable = read_language_table(config['locations']['language-table'])
    caps = config['parameters'].get('mention-capitalization', 'default')
    for infile in infiles:
        doc = FrekiDoc.read(infile)
        lgmentions = list(language_mentions(doc, lgtable, caps))
        for m in lgmentions:
            print('\t'.join(map(str, m)))


def get_instances(infiles, config):
    """
    Read Freki documents from *infiles* and return training instances

    Args:
        infiles: iterable of Freki file paths
        config: model parameters
    Yields:
        training/test instances from Freki documents
    """
    locs = config['locations']
    lgtable, olm, clm = {}, None, None
    if locs['language-table']:
        lgtable = read_language_table(locs['language-table'])
    if locs['odin-language-model']:
        olm = read_odin_language_model(locs['odin-language-model'])
    if locs['crubadan-language-model']:
        clm = read_crubadan_language_model(locs['crubadan-language-model'])

    for infile in infiles:
        doc = FrekiDoc.read(infile)

        context = {}
        context['last-lineno'] = max(x.lineno for x in doc.lines())
        caps = config['parameters'].get('mention-capitalization', 'default')
        
        lgmentions = list(language_mentions(doc, lgtable, caps))
        features_template = dict(((m.name, m.code), {}) for m in lgmentions)
        
        for span in spans(doc):
            if not span:
                continue

            context['span-top'] = span[0].lineno
            context['span-bottom'] = span[-1].lineno

            features = dict(features_template)
            gl_features(features, lgmentions, context, config)
            w_features(features, lgmentions, context, config)

            l_lines = []
            for line in span:
                context['line'] = line
                if 'L' in line.tag:
                    lgname = line.attrs.get('lang_name', '???').lower()
                    lgcode = line.attrs.get('lang_code', 'und')
                    l_feats = dict(features_template)
                    l_features(l_feats, lgmentions, olm, clm, context, config)
                    l_lines.append((line, l_feats, lgname, lgcode))
                    # if L and some other tag co-occur, only record local feats                    
                    if 'G' in line.tag:
                        g_features()
                    if 'M' in line.tag:
                        m_features(features, lgmentions, context, config)
                else:
                    # if G or M occur without L, record globally
                    if 'G' in line.tag:
                        pass
                    if 'M' in line.tag:
                        m_features(features, lgmentions, context, config)

            for l_line, l_feats, lgname, lgcode in l_lines:
                goldpair = (lgname, lgcode)
                for pair, feats in l_feats.items():
                    # print(pair, goldpair, pair == goldpair)
                    id_ = encode_instance_id(
                        os.path.splitext(os.path.basename(infile))[0],
                        l_line.span_id, l_line.lineno,
                        pair[0].replace(' ', '_'), pair[1]
                    )
                    label = True if pair == goldpair else False
                    instfeats = dict(feats)
                    instfeats.update(features[pair])
                    yield StringInstance(id_, label, instfeats)


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
    if span:
        yield span


if __name__ == '__main__':
    main()
