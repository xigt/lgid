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

import time
t1 = time.time()
t0 = time.time()

import os
from configparser import ConfigParser
import logging
import numpy as np
import re

import docopt

from freki.serialize import FrekiDoc

from lgid.models import (
    StringInstance,
    LogisticRegressionWrapper as Model,
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
    m_features,
    get_threshold_info
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
    """
    Train a language-identification model from training data

    Args:
        infiles: iterable of Freki file paths
        modelpath: the path where the model will be written
        config: model parameters
    """
    print('getting instances')
    instances = list(get_instances(infiles, config))
    # for inst in instances:
    #     print(inst.id, inst.label)
    model = Model()
    model.feat_selector = chi2
    print('training')
    model.train(instances)
    print('saving model')
    model.save(modelpath)
    get_threshold_info()
    get_feature_weights(modelpath)
    # for dist in model.test(instances):
    #     print(dist.best_class, dist.best_prob, len(dist.dict))

def find_best_and_normalize(instances, dists):
    """
        Normalize probabilities of languages and return the highest

        Args:
            instances: a list of instances relevant to a single sample of text
            dists: a list of Distributions corresponding to the instances
    """
    labels = []
    probs = []
    for i in range(len(instances)):
        lang = re.split("([0-9]+-){4}", instances[i].id)[2]
        labels.append(lang)
        assigned = bool(dists[i].best_class)
        prob = dists[i].best_prob
        if assigned:
            probs.append(prob)
        else:
            probs.append(-prob)
    probs = np.asarray(probs)
    probs = (probs - np.amin(probs)) / (np.amax(probs) - np.amin(probs))
    highest = np.argmax(probs)
    return labels[highest]


def classify(infiles, modelpath, config, instances=None):
    global t1
    """
        Classify instances found in the given files

        Args:
            infiles: iterable of Freki file paths
            modelpath: the path where the model will be loaded
            config: model parameters
            instances: a list of instances passed by test() to streamline
    """
    if not instances:
        logging.info('before getting instances: ' + str(time.time() - t1))
        t1 = time.time()
        instances = list(get_instances(infiles, config))
        logging.info('getting instances: ' + str(time.time() - t1))
        t1 = time.time()
    inst_dict = {}
    prediction_dict = {}
    for inst in instances:
        num = re.search("([0-9]+-){4}", inst.id).group(0)
        if num in inst_dict:
            inst_dict[num].append(inst)
        else:
            inst_dict[num] = [inst]
    model = Model()
    model = model.load(modelpath)
    logging.info('loading model: ' + str(time.time() - t1))
    t1 = time.time()
    for inst_id in inst_dict:
        results = model.test(inst_dict[inst_id])
        top = find_best_and_normalize(inst_dict[inst_id], results)
        prediction_dict[inst_id] = top
    logging.info('predicting:' + str(time.time() - t1))
    t1 = time.time()
    return prediction_dict

def test(infiles, modelpath, config):
    """
    Test a language-identification model

    Args:
        infiles: iterable of Freki file paths
        modelpath: the path where the model will be loaded
        config: model parameters
    """
    real_classes = {}
    instances = list(get_instances(infiles, config))
    for inst in instances:
        if bool(inst.label):
            num = re.search("([0-9]+-){4}", inst.id).group(0)
            real_classes[num] = re.split("([0-9]+-){4}", inst.id)[2]
    predicted_classes = classify(infiles, modelpath, config, instances)
    right = 0
    right_dialect = 0
    for key in real_classes:
        if key in real_classes and key in predicted_classes:
            logging.info("Language: " + real_classes[key] + '\tPredicted: ' + predicted_classes[key])
            if real_classes[key].split('-')[0] == predicted_classes[key].split('-')[0]:
                right += 1
                if real_classes[key].split == predicted_classes[key].split:
                    right_dialect += 1
    get_feature_weights(modelpath)
    print('Samples:\t' + str(len(real_classes)))
    print('Accuracy on Language (Name only):\t' + str(right / len(real_classes)))
    print('Accuracy on Dialects (Name + Code):\t' + str(right_dialect / len(real_classes)))
    print('Total time:\t' + get_time(t0))

def get_feature_weights(modelpath):
    model = Model()
    model = model.load(modelpath)
    feats_names = ["GL-first-lines", "GL-last-lines", "GL-frequent", "GL-most-frequent", "W-prev", "W-close",
                   "W-closest", "W-frequent", "W-after", "W-close-after", "W-closest-after", "W-frequent-after",
                   "L-in-line", "M-in-line", "L-LMw", "L-LMm", "L-LMc", "G-overlap", "L-CR-LMw", "L-CR-LMc",
                   "W-prevclass"]
    logging.info("Features not used:")
    for feat in feats_names:
        if feat not in model.feat_names():
            print(feat)
    logging.info("Feature weights:")
    for i in range(len(model.feat_names())):
        logging.info(model.feat_names()[i] + ": " + str(model.learner.coef_[0][i]))

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
    global t1
    """
    Read Freki documents from *infiles* and return training instances

    Args:
        infiles: iterable of Freki file paths
        config: model parameters
    Yields:
        training/test instances from Freki documents
    """
    locs = config['locations']
    lgtable = {}
    if locs['language-table']:
        lgtable = read_language_table(locs['language-table'])
        logging.info('reading lang table: ' + str(time.time() - t1))
        t1 = time.time()
    i = 1
    for infile in infiles:
        logging.info('File ' + str(i) + '/' + str(len(infiles)))
        i += 1
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
                    l_features(l_feats, lgmentions, context, config)
                    t1 = time.time()
                    l_lines.append((line, l_feats, lgname, lgcode))
                    # if L and some other tag co-occur, only record local feats
                    if 'G' in line.tag:
                        g_features(features, None, context, config)

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
