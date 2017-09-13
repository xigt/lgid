#!/usr/bin/env python3

USAGE = '''
Usage:
  lgid [-v...] train    --model=PATH [--vectors=DIR] CONFIG INFILE...
  lgid [-v...] test     --model=PATH [--vectors=DIR] CONFIG INFILE...
  lgid [-v...] classify --model=PATH --out=PATH [--vectors=DIR] CONFIG INFILE...
  lgid [-v...] list-model-weights   --model=PATH    CONFIG
  lgid [-v...] list-mentions          CONFIG INFILE...
  lgid [-v...] find-common-codes      CONFIG INFILE...
  lgid [-v...] download-crubadan-data CONFIG
  lgid [-v...] build-odin-lm          CONFIG


Commands:
  train                     train a model from supervised data
  test                      test on new data using a saved model
  classify                  output predictions on new data using a saved model
  list-mentions             just print language mentions from input files
  find-common-codes         build the text file at most-common-codes showing the most common code for each language
  list-model-weights        show feature weights in a model and features not used
  download-crubadan-data    fetch the Crubadan language model data from the web

Arguments:
  CONFIG                    path to a config file
  INFILE                    a Freki-formatted text file

Options:
  -h, --help                print this usage and exit
  -v, --verbose             increase logging verbosity
  --model PATH              where to save/load a trained model
  --out PATH                where to write freki files with added information
  --vectors DIR             a directory to print feature vectors for inspection

Examples:
  lgid -v train --model=model.gz config.ini 123.freki 456.freki
  lgid -v test --model=model.gz config.ini 789.freki
  lgid -v classify --model=model.gz config.ini 1000.freki
  lgid -v list-mentions config.ini 123.freki
  lgid -v download-crubadan-data config.ini
'''

import time
t0 = time.time()

import os
import errno
import shutil
from configparser import ConfigParser
import logging
import numpy as np
import re
import codecs

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
    decode_instance_id,
    read_crubadan_language_model,
    read_odin_language_model,
    spans,
    find_common_codes
)
from lgid.analyzers import (
    language_mentions,
)
from lgid.features import (
    gl_features,
    w_features,
    l_features,
    g_features,
    t_features,
    m_features,
    get_threshold_info,
    get_mention_by_lines
)


def main():
    args = docopt.docopt(USAGE)
    logging.basicConfig(level=50 - ((args['--verbose'] + 2) * 10))

    config = ConfigParser()
    config.read(args['CONFIG'])

    modelpath = args['--model']
    vector_dir = args['--vectors']
    if vector_dir != None:
        vector_dir = vector_dir.strip('/')
    infiles = args['INFILE']

    if args['train']:
        train(infiles, modelpath, vector_dir, config)
    elif args['classify']:
        output = args['--out']
        predictions = classify(infiles, modelpath, config, vector_dir)
        write_to_files(infiles, predictions, output)
    elif args['test']:
        test(infiles, modelpath, vector_dir, config)
    elif args['list-model-weights']:
        get_feature_weights(modelpath, config)
    elif args['list-mentions']:
        list_mentions(infiles, config)
    elif args['find-common-codes']:
        find_common_codes(infiles, config)
    elif args['download-crubadan-data']:
        download_crubadan_data(config)
    elif args['build-odin-lm']:
        build_odin_lm(config)


def write_to_files(infiles, predictions, output):
    """
    Modify freki files to include predicted language names and write to an output directory
    :param infiles: list of freki filepaths
    :param predictions: dictionary of instance-id to language name and code prediction
    :param output: filepath of the output directory
    :return: none
    """
    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output)
    for file in infiles:
        doc = FrekiDoc.read(file)
        f_name = file.split('/')[-1]
        f_name = re.sub('.freki', '', f_name)
        for span in spans(doc):
            l_lines = []
            for line in span:
                if 'L' in line.tag:
                    l_lines.append(line)
            for l_line in l_lines:
                key = (str(f_name), l_line.span_id, l_line.lineno)
                pred = predictions[key].split('-')
                lang_name = pred[0].title()
                lang_code = pred[1]
                for line in span:
                    if line.lineno >= l_line.lineno:
                        line.attrs['lang_code'] = lang_code
                        line.attrs['lang_name'] = lang_name
                        doc.set_line(line.lineno, line)
        path = output + '/' + '/'.join(file.split('/')[-2:])
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        codecs.open(path, 'w', encoding='utf8').write(str(doc))


def train(infiles, modelpath, vector_dir, config):
    """
    Train a language-identification model from training data

    Args:
        infiles: iterable of Freki file paths
        modelpath: the path where the model will be written
        vector_dir: directory where feature vectors will be written
        config: model parameters
    """
    logging.info('Getting instances')
    instances = list(get_instances(infiles, config, vector_dir))
    model = Model()
    model.feat_selector = chi2
    logging.info('Training')
    model.train(instances)
    logging.info('Saving model')
    model.save(modelpath)
    get_threshold_info()


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
        lang = '-'.join(decode_instance_id(instances[i])[-2:])
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


def classify(infiles, modelpath, config, vector_dir, instances=None):
    """
        Classify instances found in the given files

        Args:
            infiles: iterable of Freki file paths
            modelpath: the path where the model will be loaded
            config: model parameters
            instances: a list of instances passed by test() to streamline
    """
    if not instances:
        instances = list(get_instances(infiles, config, vector_dir))

    inst_dict = {}
    prediction_dict = {}
    for inst in instances:
        num = tuple(decode_instance_id(inst)[:-2])
        if num in inst_dict:
            inst_dict[num].append(inst)
        else:
            inst_dict[num] = [inst]
    model = Model()
    model = model.load(modelpath)
    for inst_id in inst_dict:
        results = model.test(inst_dict[inst_id])
        top = find_best_and_normalize(inst_dict[inst_id], results)
        prediction_dict[inst_id] = top
    return prediction_dict


def test(infiles, modelpath, vector_dir, config):
    """
    Test a language-identification model

    Args:
        infiles: iterable of Freki file paths
        modelpath: the path where the model will be loaded
        vector_dir: directory where feature vectors will be written
        config: model parameters
    """
    real_classes = {}
    instances = list(get_instances(infiles, config, vector_dir))
    for inst in instances:
        if bool(inst.label):
            num = tuple(decode_instance_id(inst)[:-2])
            real_classes[num] = '-'.join(decode_instance_id(inst)[-2:])
    predicted_classes = classify(infiles, modelpath, config, vector_dir, instances)
    right = 0
    right_dialect = 0
    right_code = 0
    for key in real_classes:
        if key in real_classes and key in predicted_classes:
            if real_classes[key].split('-')[1] == predicted_classes[key].split('-')[1]:
                right_code += 1
            if real_classes[key].split('-')[0] == predicted_classes[key].split('-')[0]:
                right += 1
                if real_classes[key] == predicted_classes[key]:
                    right_dialect += 1
    print('Samples:\t' + str(len(real_classes)))
    acc_lang = right / len(real_classes)
    acc_both = right_dialect / len(real_classes)
    acc_code = right_code / len(real_classes)
    print('Samples:\t' + str(len(real_classes)))
    print('Accuracy on Language (Name only):\t' + str(acc_lang))
    print('Accuracy on Dialects (Name + Code):\t' + str(acc_both))
    print('Accuracy on Code Only:\t' + str(acc_code))


def get_feature_weights(modelpath, config):
    """
    print config features not used and weights of individual features
    :param modelpath: path to model
    :param config: config object
    :return: none, prints to console
    """
    model = Model()
    model = model.load(modelpath)
    print("Features not used:")
    lower_feats = []
    for a_feat in model.feat_names():
        lower_feats.append(a_feat.lower())
    for feat in config['features']:
        if config['features'][feat] == 'yes':
            if str(feat) not in lower_feats:
                print('\t' + feat)
    print("Feature weights:")
    for i in range(len(model.feat_names())):
        print('\t' + model.feat_names()[i] + ": " + str(model.learner.coef_[0][i]))


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


def print_feature_vector(_id, feats, file):
    """
    print the feature values of a given vector to a file
    :param _id: instance id
    :param feats: feature dictionary
    :param file: file to write to
    :return: none, writes to file
    """
    file.write('{}: {}\n'.format(_id, ", ".join(feats)))


def get_instances(infiles, config, vector_dir):
    """
    Read Freki documents from *infiles* and return training instances

    Args:
        infiles: iterable of Freki file paths
        config: model parameters
    Yields:
        training/test instances from Freki documents
    """
    vector_file = None
    global t1
    locs = config['locations']
    lgtable = {}
    if locs['language-table']:
        lgtable = read_language_table(locs['language-table'])
    common_table = {}
    if locs['most-common-codes']:
        common_table = read_language_table(locs['most-common-codes'])
    if locs['english-word-names']:
        eng_words = open(locs['english-word-names'], 'r').read().split('\n')

    i = 1
    for infile in infiles:
        logging.info('File ' + str(i) + '/' + str(len(infiles)))
        if vector_dir != None:
            os.makedirs(vector_dir, exist_ok=True)
            vector_file = open(vector_dir + '/' + os.path.basename(infile) + '.vector', 'w')

        i += 1
        doc = FrekiDoc.read(infile)

        context = {}
        context['last-lineno'] = max(x.lineno for x in doc.lines())
        caps = config['parameters'].get('mention-capitalization', 'default')

        lgmentions = list(language_mentions(doc, lgtable, caps))
        if not lgmentions:
            lgmentions = []
        mention_dict = get_mention_by_lines(lgmentions)

        features_template = dict(((m.name, m.code), {}) for m in lgmentions)

        name_code_pairs = list(features_template.keys())
        word_clm = read_crubadan_language_model(name_code_pairs, config, False)
        char_clm = read_crubadan_language_model(name_code_pairs, config, True)
        word_olm = read_odin_language_model(name_code_pairs, config, False)
        char_olm = read_odin_language_model(name_code_pairs, config, True)
        lms = (word_clm, char_clm, word_olm, char_olm)

        for span in spans(doc):
            if not span:
                continue

            context['span-top'] = span[0].lineno
            context['span-bottom'] = span[-1].lineno

            features = dict(((m.name, m.code), {}) for m in lgmentions)

            l_lines = []
            for line in span:
                context['line'] = line
                if 'L' in line.tag:
                    lgname = line.attrs.get('lang_name', '???').lower()
                    lgcode = line.attrs.get('lang_code', 'und')
                    l_feats = dict(((m.name, m.code), {}) for m in lgmentions)
                    l_features(l_feats, mention_dict, context, lms, config)
                    t1 = time.time()
                    l_lines.append((line, l_feats, lgname, lgcode))
                    # if L and some other tag co-occur, only record local feats
                    if 'G' in line.tag:
                        g_features(features, mention_dict, context, config)
                    if 'T' in line.tag:
                        t_features(features, mention_dict, context, config)
                    if 'M' in line.tag:
                        m_features(features, mention_dict, context, config)

                else:
                    # if G, L, or M occur without L, record globally
                    if 'G' in line.tag:
                        g_features(features, mention_dict, context, config)
                    if 'T' in line.tag:
                        t_features(features, mention_dict, context, config)
                    if 'M' in line.tag:
                        m_features(features, mention_dict, context, config)

            gl_features(features, mention_dict, context, config, common_table, eng_words)
            w_features(features, mention_dict, context, config)
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
                    if vector_dir != None:
                        print_feature_vector(id_, instfeats, vector_file)
                    yield StringInstance(id_, label, instfeats)

        if vector_file:
            vector_file.close()


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
