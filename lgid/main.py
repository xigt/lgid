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


def calc_mention_recall(infiles, config):
    """
    Calculate the upper bound for language mentions: the percentage of correct labels that are mentioned in the file
    :param infiles: a list of freki filepaths
    :param config: a config object
    :return: none
    """
    lgtable = read_language_table(config['locations']['language-table'])
    caps = config['parameters'].get('mention-capitalization', 'default')
    positive = 0
    length = 0
    index = 1
    for file in infiles:
        doc = FrekiDoc.read(file)
        mentions = language_mentions(doc, lgtable, caps)
        texts, labels = get_instances([file])
        print(str(index) + '/' + str(len(infiles)))
        index += 1
        length += len(labels)
        names = set()
        for mention in mentions:
            name = mention.name[0].upper() + mention.name[1:] + '_' + mention.code
            names.add(name)
        for label in labels:
            if label in names:
                positive += 1
    print("Language mention recall: " + str(float(positive)/length))




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


def train(infiles, modelpath, config):
    global t1
    if os.path.exists(modelpath):
        shutil.rmtree(modelpath)
    os.makedirs(modelpath)
    texts, labels = get_instances(infiles)
    print('Getting instances: ' + str(time.time() - t1))
    t1 = time.time()
    char_max = int(config['parameters']['character-n-gram-size'])
    char_count = Vectorizer(texts, ngram_range=(1, char_max), analyzer='char').fit(texts)
    word_max = int(config['parameters']['word-n-gram-size'])
    word_count = Vectorizer(texts, ngram_range=(1, word_max), analyzer='word').fit(texts)
    print('Training vectorizers: ' + str(time.time() - t1))
    t1 = time.time()
    char_matrix = char_count.transform(texts)
    word_matrix = word_count.transform(texts)
    pickle.dump(char_count, open(modelpath + '/char.p', 'wb'))
    pickle.dump(word_count, open(modelpath + '/word.p', 'wb'))
    main_x = hstack([char_matrix, word_matrix])
    labels = labels
    model = Model()
    model.fit(main_x, labels)
    print('Fitting model: ' + str(time.time() - t1))
    pickle.dump(model, open(modelpath + '/model.p', 'wb'))
    t1 = time.time()


def classify(texts, modelpath, config):
    model = pickle.load(open(modelpath + '/model.p', 'rb'))
    char_counts = pickle.load(open(modelpath + '/char.p', 'rb'))
    word_counts = pickle.load(open(modelpath + '/word.p', 'rb'))

    char_matrix = char_counts.transform(texts)
    word_matrix = word_counts.transform(texts)

    main_x = hstack([char_matrix, word_matrix])
    result = model.predict(main_x)
    return result

def test(infiles, modelpath, config):
    print('Testing')
    texts, y = get_instances(infiles)

    result = classify(texts, modelpath, config)

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
    print('Total time:\t' + str(time.time() - t0))


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


def get_instances(infiles):
    texts = []
    labels = []
    instances = []
    for file in infiles:
        for line in open(file, 'r').readlines():
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
                instances.append((text, label))
    return texts, labels


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
