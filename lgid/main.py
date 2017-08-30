#!/usr/bin/env python3

USAGE = '''
Usage:
  lgid [-v...] train    --model=PATH [--vectors=DIR] CONFIG INFILE...
  lgid [-v...] test     --model=PATH [--vectors=DIR] CONFIG INFILE...
  lgid [-v...] validate     --model=PATH [--vectors=DIR] CONFIG INFILE...
  lgid [-v...] classify --model=PATH --out=PATH [--vectors=DIR] CONFIG INFILE...
  lgid [-v...] get-lg-recall    CONFIG INFILE...
  lgid [-v...] list-model-weights   --model=PATH    CONFIG
  lgid [-v...] list-mentions          CONFIG INFILE...
  lgid [-v...] download-crubadan-data CONFIG
  lgid [-v...] build-odin-lm          CONFIG


Commands:
  train                     train a model from supervised data
  test                      test on new data using a saved model
  validate                  Perform n-fold cross validation on the data
  classify                  output predictions on new data using a saved model
  get-lg-recall             find the language mention recall upper bound for a set of files
  list-mentions             just print language mentions from input files
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
t1 = time.time()
t0 = time.time()

import os
import shutil
from configparser import ConfigParser
import logging
import numpy as np
import re
from random import randint
from collections import namedtuple


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
    read_odin_language_model
)
from lgid.analyzers import (
    language_mentions,
    Mention
)
from lgid.features import (
    gl_features,
    w_features,
    l_features,
    g_features,
    m_features,
    get_threshold_info
)

instance_dict = {}

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
    elif args['get-lg-recall']:
        calc_mention_recall(infiles, config)
    elif args['test']:
        test(infiles, modelpath, vector_dir, config)
    elif args['validate']:
        n_fold_validation(5, infiles, modelpath, vector_dir, config)
    elif args['list-model-weights']:
        get_feature_weights(modelpath, config)
    elif args['list-mentions']:
        list_mentions(infiles, config)
    elif args['download-crubadan-data']:
        download_crubadan_data(config)
    elif args['build-odin-lm']:
        build_odin_lm(config)


def calc_mention_recall(infiles, config, instances=None):
    """
    Calculate the upper bound for language mentions: the percentage of correct labels that are mentioned in the file
    :param infiles: a list of freki filepaths
    :param config: a config object
    :return: none
    """
    if not instances:
        instances = list(get_instances(infiles, config, None))
    lgtable = read_language_table(config['locations']['language-table'])
    caps = config['parameters'].get('mention-capitalization', 'default')
    positive = 0
    length = 0
    file_dict = {}
    for inst in instances:
        if inst.label:
            file_num = inst.id.split('-')[0]
            lang = '-'.join(inst.id.split('-')[-2:])
            if file_num in file_dict:
                file_dict[file_num].append(lang)
            else:
                file_dict[file_num] = [lang]
    for file in infiles:
        doc = FrekiDoc.read(file)
        num = doc.get_line(1).block.doc_id
        if num in file_dict:
            mentions = language_mentions(doc, lgtable, caps)
            length += len(file_dict[num])
            for label in file_dict[num]:
                n = label.split('-')[0]
                c = label.split('-')[1]
                for mention in mentions:
                    if n == mention.name and c == mention.code:
                        positive += 1
                        break
        else:
            print(num)
            print(file_dict.keys())

    recall = float(positive)/length
    print("Language mention recall: " + str(recall))
    return recall

def n_fold_validation(n, infiles, modelpath, vector_dir, config):
    accs_lang = []
    accs_both = []
    accs_code = []
    recalls = []
    groups = {}
    for file in infiles:
        choice = randint(1, n)
        if choice in groups:
            groups[choice].append(file)
        else:
            groups[choice] = [file]
    i = 1
    for group in groups:
        logging.info("Cross validation group " + str(i) + '/' + str(n))
        i += 1
        training = []
        testing = groups[group]
        test_inst = list(get_instances(testing, config, vector_dir))
        recall = calc_mention_recall(testing, config, instances=test_inst)
        for g2 in groups:
            if g2 != group:
                training.extend(groups[g2])
        train(training, modelpath, vector_dir, config)
        acc_lang, acc_both, acc_code = test(testing, modelpath, vector_dir, config, instances=test_inst)
        accs_lang.append(acc_lang)
        accs_both.append(acc_both)
        accs_code.append(acc_both)
        recalls.append(recall)
    print('Average and Std Dev of:')
    print('Language Only:\t' + str(np.average(accs_lang)) + '\t' + str(np.std(accs_lang)))
    print('Language and Code:\t' + str(np.average(accs_both)) + '\t' + str(np.std(accs_both)))
    print('Code Only:\t' + str(np.average(accs_code)) + '\t' + str(np.std(accs_code)))
    print('Language Mention Recall:\t' + str(np.average(recalls)) + '\t' + str(np.std(recalls)))


def get_time(t):
    m, s = divmod(time.time() - t, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


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
        number = None
        span_dict = doc.spans()
        for span in span_dict:
            start, end = span_dict[span]
            start_line = doc.get_line(start)
            number = start_line.block.doc_id
            key = number + "-" + start_line.span_id + '-' + str(start_line.lineno) + '-'
            pred = predictions[key].split('-')
            lang_name = pred[0].title()
            lang_code = pred[1]
            for i in range(start, end):
                line = doc.get_line(i)
                line.attrs['lang_code'] = lang_code
                line.attrs['lang_name'] = lang_name
                doc.set_line(i, line)
        open(output + '/' + str(number) + '.freki', 'w').write(str(doc))

def train(infiles, modelpath, vector_dir, config, instances=None):
    """
    Train a language-identification model from training data

    Args:
        infiles: iterable of Freki file paths
        modelpath: the path where the model will be written
        config: model parameters
    """
    if not instances:
        print('getting instances')
        instances = list(get_instances(infiles, config, vector_dir))
    model = Model()
    model.feat_selector = chi2
    print('training')
    model.train(instances)
    print('saving model')
    model.save(modelpath)
    get_threshold_info()
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


def classify(infiles, modelpath, config, vector_dir, instances=None):
    """
        Classify instances found in the given files

        Args:
            infiles: iterable of Freki file paths
            modelpath: the path where the model will be loaded
            config: model parameters
            instances: a list of instances passed by test() to streamline
    """
    global t1
    if not instances:
        t1 = time.time()
        instances = list(get_instances(infiles, config, vector_dir))
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
    t1 = time.time()
    for inst_id in inst_dict:
        results = model.test(inst_dict[inst_id])
        top = find_best_and_normalize(inst_dict[inst_id], results)
        prediction_dict[inst_id] = top
    t1 = time.time()
    return prediction_dict

def test(infiles, modelpath, vector_dir, config, instances=None):
    """
    Test a language-identification model

    Args:
        infiles: iterable of Freki file paths
        modelpath: the path where the model will be loaded
        config: model parameters
    """
    real_classes = {}
    if not instances:
        instances = list(get_instances(infiles, config, vector_dir))
    for inst in instances:
        if bool(inst.label):
            num = re.search("([0-9]+-){4}", inst.id).group(0)
            real_classes[num] = re.split("([0-9]+-){4}", inst.id)[2]
    predicted_classes = classify(infiles, modelpath, config, vector_dir, instances=instances)
    right = 0
    right_dialect = 0
    right_code = 0
    file_counts = {}
    mistake_counts = {}
    for key in real_classes:
        key2 = key.split('-')[0]
        if key2 not in file_counts:
            file_counts[key2] = [0, 0]
        file_counts[key2][1] += 1
        if key in predicted_classes:
            if real_classes[key].split('-')[1] == predicted_classes[key].split('-')[1]:
                right_code += 1
            if real_classes[key].split('-')[0] == predicted_classes[key].split('-')[0]:
                right += 1
                if real_classes[key] == predicted_classes[key]:
                    right_dialect += 1
                    file_counts[key2][0] += 1
            else:
                mistake_key = (real_classes[key], predicted_classes[key])
                if mistake_key in mistake_counts:
                    mistake_counts[mistake_key] += 1
                else:
                    mistake_counts[mistake_key] = 1
    for key2 in file_counts:
        logging.info("Accuracy on file " + str(key2) + ':\t' + str(float(file_counts[key2][0]) / file_counts[key2][1]))
    mistakes = open('errors.txt', 'w')
    mistakes.write('(real, predicted)\tcount\n')
    for mistake_key in sorted(mistake_counts, key=lambda x: mistake_counts[x], reverse=True):
        mistakes.write(str(mistake_key) + '\t' + str(mistake_counts[mistake_key]) + '\n')
    acc_lang = right / len(real_classes)
    acc_both = right_dialect / len(real_classes)
    acc_code = right_code / len(real_classes)
    print('Samples:\t' + str(len(real_classes)))
    print('Accuracy on Language (Name only):\t' + str(acc_lang))
    print('Accuracy on Dialects (Name + Code):\t' + str(acc_both))
    print('Accuracy on Code Only:\t' + str(acc_code))
    print('Total time:\t' + get_time(t0))
    return acc_lang, acc_both, acc_code


def get_feature_weights(modelpath, config):
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
    file.write('{}: {}\n'.format(_id, ", ".join(feats)))

def get_instances(infiles, config, vector_dir):
    insts = []
    index = 1
    for file in infiles:
        logging.info("Instances from file " + str(index) + '/' + str(len(infiles)))
        index += 1
        if file not in instance_dict:
            instance_dict[file] = list(real_get_instances([file], config, vector_dir))
        insts.extend(instance_dict[file])
    return insts

def real_get_instances(infiles, config, vector_dir):
    vector_file = None
    """
    Read Freki documents from *infiles* and return training instances

    Args:
        infiles: iterable of Freki file paths
        config: model parameters
    Yields:
        training/test instances from Freki documents
    """
    global t1
    locs = config['locations']
    lgtable = {}
    if locs['language-table']:
        lgtable = read_language_table(locs['language-table'])
    i = 1
    for infile in infiles:
        #logging.info('File ' + str(i) + '/' + str(len(infiles)))
        if vector_dir != None:
            os.makedirs(vector_dir, exist_ok=True)
            vector_file = open(vector_dir + '/' + os.path.basename(infile) + '.vector', 'w')

        i += 1
        doc = FrekiDoc.read(infile)

        context = {}
        context['last-lineno'] = max(x.lineno for x in doc.lines())
        caps = config['parameters'].get('mention-capitalization', 'default')

        lgmentions = list(language_mentions(doc, lgtable, caps))

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
            gl_features(features, lgmentions, context, config)
            w_features(features, lgmentions, context, config)
            l_lines = []
            for line in span:
                context['line'] = line
                if 'L' in line.tag:
                    lgname = line.attrs.get('lang_name', '???').lower()
                    lgcode = line.attrs.get('lang_code', 'und')
                    l_feats = dict(((m.name, m.code), {}) for m in lgmentions)
                    l_features(l_feats, lgmentions, context, lms, config)
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
                    if vector_dir != None:
                        print_feature_vector(id_, instfeats, vector_file)
                    yield StringInstance(id_, label, instfeats)
        if vector_file:
            vector_file.close()


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
