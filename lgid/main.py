#!/usr/bin/env python3

USAGE = '''
Usage:
  lgid [-v...] train    --model=PATH  CONFIG INFILE...
  lgid [-v...] classify --model=PATH  CONFIG INFILE...
  lgid [-v...] test     --model=PATH  CONFIG INFILE...
  lgid [-v...] list-mentions          CONFIG INFILE...
  lgid [-v...] download-crubadan-data CONFIG
  lgid [-v...] build-odin-lm          CONFIG


Commands:
  train                     train a model from supervised data
  test                      test on new data using a saved model
  classify                  output predictions on new data using a saved model
  list-mentions             just print language mentions from input files
  download-crubadan-data    fetch the Crubadan language model data from the web

Arguments:
  CONFIG                    path to a config file
  INFILE                    a Freki-formatted text file

Options:
  -h, --help                print this usage and exit
  -v, --verbose             increase logging verbosity
  --model PATH              where to save/load a trained model

Examples:
  lgid -v train --model=model.gz config.ini 123.freki 456.freki
  lgid -v test --model=model.gz config.ini 789.freki
  lgid -v classify --model=model.gz config.ini 1000.freki
  lgid -v list-mentions config.ini 123.freki
  lgid -v download-crubadan-data config.ini
'''

import os
from configparser import ConfigParser
import logging
import numpy as np
import re

import docopt

from freki.serialize import FrekiDoc

from sklearn.feature_extraction.text import CountVectorizer as Vectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier as Model
from scipy.sparse import hstack
import shutil
import time
import pickle
t0 = time.time()
t1 = time.time()

from lgid.models import (
    StringInstance,
    chi2
)

from lgid.util import (
    read_language_table,
    encode_instance_id,
    decode_instance_id
)
from lgid.analyzers import (
    language_mentions
)
from lgid.features import (
    gl_features,
    w_features,
    l_features,
    g_features,
    m_features
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
    elif args['classify']:
        classify(infiles, modelpath, config)
    elif args['test']:
        test(infiles, modelpath, config)
    elif args['list-mentions']:
        list_mentions(infiles, config)
    elif args['download-crubadan-data']:
        download_crubadan_data(config)
    elif args['build-odin-lm']:
        build_odin_lm(config)


def get_time(t):
    m, s = divmod(time.time() - t, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def train(infiles, modelpath, config):
    global t0
    global t1
    if os.path.exists(modelpath):
        shutil.rmtree(modelpath)
    os.makedirs(modelpath)
    texts, labels, others = get_instances(infiles, config)
    print('Getting instances: ' + str(time.time() - t1))
    t1 = time.time()
    char_max = int(config['parameters']['character-n-gram-size'])
    char_count = Vectorizer(texts, ngram_range=(1, char_max), analyzer='char').fit(texts)
    word_max = int(config['parameters']['word-n-gram-size'])
    word_count = Vectorizer(texts, ngram_range=(1, word_max), analyzer='word').fit(texts)
    feat_vectr = DictVectorizer().fit(others)
    print('Training vectorizers: ' + str(time.time() - t1))
    t1 = time.time()
    char_matrix = char_count.transform(texts)
    word_matrix = word_count.transform(texts)
    other_matrix = feat_vectr.transform(others)
    pickle.dump(char_count, open(modelpath + '/char.p', 'wb'))
    pickle.dump(word_count, open(modelpath + '/word.p', 'wb'))
    pickle.dump(feat_vectr, open(modelpath + '/feats.p', 'wb'))
    main_x = hstack([char_matrix, word_matrix, other_matrix])
    labels = labels
    model = Model()
    model.fit(main_x, labels)
    print('Fitting model: ' + str(time.time() - t1))
    pickle.dump(model, open(modelpath + '/model.p', 'wb'))
    t1 = time.time()
    print("Total training: " + get_time(t0))


def classify(texts, others, modelpath, config):
    model = pickle.load(open(modelpath + '/model.p', 'rb'))
    char_counts = pickle.load(open(modelpath + '/char.p', 'rb'))
    word_counts = pickle.load(open(modelpath + '/word.p', 'rb'))
    feat_vectr = pickle.load(open(modelpath + '/feats.p', 'rb'))

    char_matrix = char_counts.transform(texts)
    word_matrix = word_counts.transform(texts)
    other_matrix = feat_vectr.transform(others)

    main_x = hstack([char_matrix, word_matrix, other_matrix])
    result = model.predict(main_x)
    return result

def test(infiles, modelpath, config):
    print('Testing')
    texts, y, others = get_instances(infiles, config)

    result = classify(texts, others, modelpath, config)

    error_dict = {}
    error = open('errors.txt', 'w')

    right = 0
    for i in range(len(y)):
        if y[i] == result[i]:
            right += 1
        else:
            key = 'actual:\t' + y[i] + '\npredicted:\t' + result[i] + '\n'
            if key in error_dict:
                error_dict[key] += 1
            else:
                error_dict[key] = 1
    for key in error_dict:
        error.write(key)
        error.write('count:\t' + str(error_dict[key]) + '\n')
        error.write('\n')

    print('Samples:\t' + str(len(y)))
    print('Accuracy:\t' + str(right / len(y)))
    print('Total time:\t' + get_time(t0))


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

def get_mention_dict(mentions):
    """
    Get a dictionary mapping line number to a list of language mentions

    :param mentions: list of language mentions
    :return: dictionary of (int) line number to list of mentions
    """
    mention_dict = {}
    for mention in mentions:
        index = int(mention.startline)
        mention_dict[index] = [mention]
        if index in mention_dict:
            mention_dict[index].append(mention)
        else:
            mention_dict[index] = [mention]
    return mention_dict

def get_features(line_num, mention_dict, config):
    """
    Find mention features relevant to language text on a given line. features are in the form of:
    number of times a language is mentioned in the window -- (lang_name)_w : int
    number of times a language is mentioned in the close window -- (lang_name)_c : int

    :param line_num: line number of the language text to be identified
    :param mention_dict: dictionary of line_number to list of mentions
    :param config: config object
    :return: dictionary of feature to count
    """
    before = int(config['parameters'].get('window-size', 'default'))
    after = int(config['parameters'].get('after-window-size', 'default'))
    c_before = int(config['parameters'].get('close-window-size', 'default'))
    c_after = int(config['parameters'].get('after-close-window-size', 'default'))
    feat_dict = {}
    for i in range(line_num - before, line_num + after):
        if i in mention_dict:
            mentions = mention_dict[i]
            for mention in mentions:
                lang = mention.name
                key = lang + '_w'
                if key in feat_dict:
                    feat_dict[key] += 1
                else:
                    feat_dict[key] = 1
                if line_num - c_before < i < line_num + c_after:
                    key = lang + '_c'
                    if key in feat_dict:
                        feat_dict[key] += 1
                    else:
                        feat_dict[key] = 1
    return feat_dict

def get_instances(infiles, config):
    """
    Find all instances of language text to be identified

    :param infiles: list of freki file paths
    :param config: config object
    :return: texts: list of language strings,
            labels: list of gold labels for the texts,
            others: list of dicts of mention features for each text
    """
    texts = []
    labels = []
    other_feats = []
    lgtable = read_language_table(config['locations']['language-table'])
    caps = config['parameters'].get('mention-capitalization', 'default')
    j = 1
    for file in infiles:
        print(str(j) + '/' + str(len(infiles)))
        j += 1
        doc = FrekiDoc.read(file)
        glob_mentions = {}
        lgmentions = list(language_mentions(doc, lgtable, caps))
        lg_dict = get_mention_dict(lgmentions)
        for mention in lgmentions:
            lang = mention.name
            if lang in glob_mentions:
                glob_mentions[lang] += 1
            else:
                glob_mentions[lang] = 1
        all_text = open(file, 'r').read()
        lines = all_text.split('\n')
        for i in range(len(lines)):
            line = lines[i]
            if "tag=L" in line:
                lang = re.search('lang_name=.+lang_code', line).group(0).split('=')[1].split('lang_code')[0].strip()
                lang_code = re.search('lang_code=.+fonts', line).group(0).split('=')[1].split('fonts')[0].strip()
                label = lang + '_' + lang_code
                text = line.split(':')[1]
                if type(text) == 'list':
                    text = ''.join(text)
                text = text.strip()
                text = re.sub('\s+', ' ', text)
                texts.append(text)
                labels.append(label)
                feat_dict = get_features(i + 1, lg_dict, config)
                feat_dict.update(glob_mentions)
                other_feats.append(feat_dict)
    return texts, labels, other_feats


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


def download_crubadan_data(config):
    """
    Download and extract Crubadan language data
    """
    from io import BytesIO
    from zipfile import ZipFile
    import csv
    import requests

    logging.info('Downloading Crubadan data')

    index = config['locations']['crubadan-index']
    baseuri = config['locations']['crubadan-base-uri']
    output_dir = config['locations']['crubadan-language-model']

    os.makedirs(output_dir, exist_ok=True)

    table = open(index, 'r', encoding='utf8')
    reader = csv.reader(table)

    i = j = 0
    header = next(reader)  # discard header row
    for row in reader:
        code = row[0]
        iso_code = row[8].strip()
        url = requests.compat.urljoin(baseuri, code + '.zip')
        combined_code = "{}_{}".format(iso_code, code)
        dest = os.path.join(output_dir, combined_code)

        logging.debug(
            'Downloading Crubadan data for {} from {}'.format(combined_code, url)
        )
        response = requests.get(url)
        file = ZipFile(BytesIO(response.content))
        i += 1

        # basic validation (won't cover every scenario!)
        if any(os.path.exists(os.path.join(dest, p)) for p in file.namelist()):
            logging.error(
                'Unzipping the archive for {} will overwrite data! Skipping...'
                .format(combined_code)
            )
            continue

        file.extractall(dest)
        j += 1
        logging.debug('Successfully extracted data for {}'.format(combined_code))

    logging.info('Successfully downloaded {} files from Crubadan'.format(i))
    logging.info('Successfully extracted {} files from Crubadan'.format(j))


def build_odin_lm(config):
    from lgid.buildlms import build_from_odin

    """
    Build the LMs from the odin data
    """
    logging.info('Building ODIN LMs')
    indirec = config['locations']['odin-source']
    outdirec = config['locations']['odin-language-model']
    nc = config['parameters']['character-n-gram-size']
    nw = config['parameters']['word-n-gram-size']
    build_from_odin(indirec, outdirec, nc, nw)
    logging.info('Successfully built ODIN LMs')


if __name__ == '__main__':
    main()
