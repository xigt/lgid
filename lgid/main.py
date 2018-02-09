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
  lgid [-v...] count-mentions         CONFIG INFILE...
  lgid [-v...] find-common-codes      CONFIG INFILE...
  lgid [-v...] download-crubadan-data CONFIG
  lgid [-v...] build-odin-lm          CONFIG


Commands:
  train                     train a model from supervised data
  test                      test on new data using a saved model
  validate                  Perform n-fold cross validation on the data
  classify                  output predictions on new data using a saved model
  get-lg-recall             find the language mention recall upper bound for a set of files
  list-model-weights        show feature weights in a model and features not used
  list-mentions             just print language mentions from input files
  count-mentions            count the number of mentions of each language found in the input files
  find-common-codes         build the text file at most-common-codes showing the most common code for each language
  download-crubadan-data    fetch the Crubadan language model data from the web
  build-odin-lm             produces language model files from ODIN data

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
from configparser import ConfigParser
import logging
import numpy as np
import re
import random
random.seed(1)
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
    find_common_codes,
    read_language_mapping_table
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
    get_mention_by_lines
)


def main():
    args = docopt.docopt(USAGE)
    logging.basicConfig(level=50 - ((args['--verbose'] + 2) * 10))

    config = ConfigParser()
    config.read(args['CONFIG'])

    modelpath = args['--model']
    vector_dir = args['--vectors']
    single_mention = config['parameters']['single-longest-mention'] == 'yes'
    if vector_dir != None:
        vector_dir = vector_dir.strip('/')
    infiles = args['INFILE']
    if args['train']:
        train(infiles, modelpath, vector_dir, config, single_mention)
    elif args['classify']:
        output = args['--out']
        predictions = classify(infiles, modelpath, config, vector_dir, single_mention)
        write_to_files(infiles, predictions, output)
    elif args['get-lg-recall']:
        calc_mention_recall(infiles, config, single_mention)
    elif args['test']:
        test(infiles, modelpath, vector_dir, config, single_mention)
    elif args['validate']:
        n_fold_validation(5, infiles, modelpath, vector_dir, config, single_mention)
    elif args['list-model-weights']:
        get_feature_weights(modelpath, config)
    elif args['list-mentions']:
        list_mentions(infiles, config, single_mention)
    elif args['count-mentions']:
        count_mentions(infiles, config, single_mention)
    elif args['find-common-codes']:
        find_common_codes(infiles, config)
    elif args['download-crubadan-data']:
        download_crubadan_data(config)
    elif args['build-odin-lm']:
        build_odin_lm(config)


def calc_mention_recall(infiles, config, single_mention, instances=None):
    """
    Calculate the upper bound for language mentions: the percentage of correct labels that are mentioned in the file
    :param infiles: a list of freki filepaths
    :param config: a config object
    :param single_mention: whether to use the longest mention or all mentions from each language mention span
    :return: none
    """
    if not instances:
        instances = list(get_instances(infiles, config, None, single_mention))
    lgtable = read_language_table(config['locations']['language-table'])
    caps = config['parameters'].get('mention-capitalization', 'default')
    lang_mapping_tables = read_language_mapping_table(config)
    positive = 0
    length = 0
    file_dict = {}
    for inst in instances:
        if inst.label:
            file_num = inst.id[0]
            lang = '-'.join(inst.id[-2:])
            if file_num in file_dict:
                file_dict[file_num].append(lang)
            else:
                file_dict[file_num] = [lang]
    for file in infiles:
        doc = FrekiDoc.read(file)
        num = doc.get_line(1).block.doc_id
        if num in file_dict:
            mentions = list(language_mentions(doc, lgtable, lang_mapping_tables, caps, single_mention))
            length += len(file_dict[num])
            for label in file_dict[num]:
                n = label.split('-')[0]
                n = re.sub('_', ' ', n)
                c = label.split('-')[1]
                for mention in mentions:
                    if n == mention.name and c == mention.code:
                        positive += 1
                        break

    if length:
        recall = float(positive)/length
    else:
        recall = 1
    print("Language mention recall: " + str(recall))
    return recall

mistake_counts = {}
lang_accs = {}
lm_sizes = {}
file_accs = {}
file_mentions = {}

def n_fold_validation(n, infiles, modelpath, vector_dir, config, single_mention):
    instance_dict = {}
    accs_lang = []
    accs_both = []
    accs_code = []
    recalls = []
    groups = {}
    for file in infiles:
        choice = random.randint(1, n)
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
        for g2 in groups:
            if g2 != group:
                training.extend(groups[g2])
        train_data = list(cached_get_instances(training, config, vector_dir, instance_dict, single_mention))
        train(training, modelpath, vector_dir, config, single_mention, instances=train_data)
        test_inst = list(cached_get_instances(testing, config, vector_dir, instance_dict, single_mention))
        recall = calc_mention_recall(testing, config, single_mention, instances=test_inst)
        acc_lang, acc_both, acc_code = test(testing, modelpath, vector_dir, config, single_mention, instances=test_inst)
        accs_lang.append(acc_lang)
        accs_both.append(acc_both)
        accs_code.append(acc_both)
        recalls.append(recall)
    print('Average and Std Dev of:')
    print('Language Only:\t' + str(np.average(accs_lang)) + '\t' + str(np.std(accs_lang)))
    print('Language and Code:\t' + str(np.average(accs_both)) + '\t' + str(np.std(accs_both)))
    print('Code Only:\t' + str(np.average(accs_code)) + '\t' + str(np.std(accs_code)))
    print('Language Mention Recall:\t' + str(np.average(recalls)) + '\t' + str(np.std(recalls)))
    x = []
    y = []
    for file in file_accs:
        file_acc = file_accs[file]
        if file in file_mentions:
            x.append(file_acc)
            y.append(file_mentions[file])
    #pr.create_stats()
    #pr.print_stats('cumtime')
    #pr.disable()


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
    os.makedirs(output, exist_ok=True)
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
        path = output + '/' + file.split('/')[-1]
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        codecs.open(path, 'w', encoding='utf8').write(str(doc))


def train(infiles, modelpath, vector_dir, config, single_mention, instances=None):
    """
    Train a language-identification model from training data

    Args:
        infiles: iterable of Freki file paths
        modelpath: the path where the model will be written
        vector_dir: directory where feature vectors will be written
        config: model parameters
    """
    if not instances:
        logging.info('Getting instances')
        instances = list(get_instances(infiles, config, vector_dir, single_mention))
    model = Model()
    model.feat_selector = chi2
    logging.info('Training')
    model.train(instances)
    logging.info('Saving model')
    model.save(modelpath)


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


def classify(infiles, modelpath, config, vector_dir, single_mention, instances=None):
    """
        Classify instances found in the given files

        Args:
            infiles: iterable of Freki file paths
            modelpath: the path where the model will be loaded
            config: model parameters
            instances: a list of instances passed by test() to streamline
    """
    if not instances:
        instances = list(get_instances(infiles, config, vector_dir, single_mention))

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


def test(infiles, modelpath, vector_dir, config, single_mention, instances=None):
    """
    Test a language-identification model

    Args:
        infiles: iterable of Freki file paths
        modelpath: the path where the model will be loaded
        vector_dir: directory where feature vectors will be written
        config: model parameters
    """
    real_classes = {}
    if not instances:
        instances = list(get_instances(infiles, config, vector_dir, single_mention))
    for inst in instances:
        if bool(inst.label):
            num = tuple(decode_instance_id(inst)[:-2])
            real_classes[num] = '-'.join(decode_instance_id(inst)[-2:])
    predicted_classes = classify(infiles, modelpath, config, vector_dir, instances)
    right = 0
    right_dialect = 0
    right_code = 0
    file_counts = {}
    for key in real_classes:
        key2 = key[0]
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
            if real_classes[key] != predicted_classes[key]:
                print(key)
                if real_classes[key] in lang_accs:
                    lang_accs[real_classes[key]][1] += 1
                else:
                    lang_accs[real_classes[key]] = [0, 1]
                mistake_key = (real_classes[key], predicted_classes[key])
                if mistake_key in mistake_counts:
                    mistake_counts[mistake_key] += 1
                else:
                    mistake_counts[mistake_key] = 1
            else:
                if real_classes[key] in lang_accs:
                    lang_accs[real_classes[key]][0] += 1
                    lang_accs[real_classes[key]][1] += 1
                else:
                    lang_accs[real_classes[key]] = [1, 1]
    for key2 in file_counts:
        file_acc = float(file_counts[key2][0]) / file_counts[key2][1]
        logging.info("Accuracy on file " + str(key2) + ':\t' + str(file_acc))
        file_accs[key2] = file_acc
    mistakes = open(config['locations']['classify-error-file'], 'w')
    mistakes.write('(real, predicted)\tcount\n')
    for mistake_key in sorted(mistake_counts, key=lambda x: mistake_counts[x], reverse=True):
        mistakes.write(str(mistake_key) + '\t' + str(mistake_counts[mistake_key]) + '\n')
    print('Samples:\t' + str(len(real_classes)))
    acc_lang = right / len(real_classes)
    acc_both = right_dialect / len(real_classes)
    acc_code = right_code / len(real_classes)
    print('Accuracy on Language (Name only):\t' + str(acc_lang))
    print('Accuracy on Dialects (Name + Code):\t' + str(acc_both))
    print('Accuracy on Code Only:\t' + str(acc_code))
    return acc_lang, acc_both, acc_code


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


def list_mentions(infiles, config, single_mention):
    """
    List all language mentions found in the given files

    Args:
        infiles: iterable of Freki file paths
        config: model parameters
    """
    lgtable = read_language_table(config['locations']['language-table'])
    lang_mapping_tables = read_language_mapping_table(config)
    caps = config['parameters'].get('mention-capitalization', 'default')
    for infile in infiles:
        doc = FrekiDoc.read(infile)
        lgmentions = list(language_mentions(doc, lgtable, lang_mapping_tables, caps, single_mention))
        for m in lgmentions:
            print('\t'.join(map(str, m)))


def count_mentions(infiles, config, single_mention):
    """
    List all languages mentioned found in the given files with the count of
    how many times each was mentioned

    Args:
        infiles: iterable of Freki file paths
        config: model parameters
    """
    lgtable = read_language_table(config['locations']['language-table'])
    lang_mapping_tables = read_language_mapping_table(config)
    caps = config['parameters'].get('mention-capitalization', 'default')
    mentions = {}
    for infile in infiles:
        doc = FrekiDoc.read(infile)
        lgmentions = list(language_mentions(doc, lgtable, lang_mapping_tables, caps, single_mention))
        for m in lgmentions:
            if m[4] in mentions:
                mentions[m[4]] += 1
            else:
                mentions[m[4]] = 1
    for m in mentions:
        mentions[m] = int(mentions[m] / len(lgtable[m]))
    ordered = sorted(mentions, key=lambda x: mentions[x], reverse=True)
    for m in ordered:
        print('{}: {}'.format(m, mentions[m]))


def print_feature_vector(_id, feats, file):
    """
    print the feature values of a given vector to a file
    :param _id: instance id
    :param feats: feature dictionary
    :param file: file to write to
    :return: none, writes to file
    """
    file.write('{}: {}\n'.format(_id, ", ".join(feats)))


def cached_get_instances(infiles, config, vector_dir, instance_dict, single_mention):
    locs = config['locations']
    lgtable = {}
    if locs['language-table']:
        lgtable = read_language_table(locs['language-table'])
    common_table = {}
    if locs['most-common-codes']:
        common_table = read_language_table(locs['most-common-codes'])
    eng_words = open(locs['english-word-names'], 'r').read().split('\n')
    insts = []
    index = 1
    for file in infiles:
        logging.info("Instances from file " + str(index) + '/' + str(len(infiles)))
        index += 1
        if file not in instance_dict:
            instance_dict[file] = list(get_instances([file], config, vector_dir, single_mention, lgtable, common_table, eng_words))
        insts.extend(instance_dict[file])
    return insts


def get_instances(infiles, config, vector_dir, single_mention, lgtable=None, common_table=None, eng_words=None):
    vector_file = None
    """
    Read Freki documents from *infiles* and return training instances

    Args:
        infiles: iterable of Freki file paths
        config: model parameters
    Yields:
        training/test instances from Freki documents
    """
    if not lgtable:
        locs = config['locations']
        lgtable = {}
        if locs['language-table']:
            lgtable = read_language_table(locs['language-table'])
        common_table = {}
        if locs['most-common-codes']:
            common_table = read_language_table(locs['most-common-codes'])
        if locs['english-word-names']:
            eng_words = open(locs['english-word-names'], 'r').read().split('\n')
    lang_mapping_tables = read_language_mapping_table(config)
    global t1
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

        lgmentions = list(language_mentions(doc, lgtable, lang_mapping_tables, caps, single_mention))
        if not lgmentions:
            lgmentions = []
        mention_dict = get_mention_by_lines(lgmentions)

        features_template = dict(((m.name, m.code), {}) for m in lgmentions)
        lang_names = set(m.name for m in lgmentions)
        file_mentions[doc.blocks[0].doc_id] = len(lang_names)

        name_code_pairs = list(features_template.keys())
        word_clm = read_crubadan_language_model(name_code_pairs, config, 'word')
        char_clm = read_crubadan_language_model(name_code_pairs, config, 'character')
        word_olm = read_odin_language_model(name_code_pairs, config, 'word')
        char_olm = read_odin_language_model(name_code_pairs, config, 'character')
        morph_olm = read_odin_language_model(name_code_pairs, config, 'morpheme')
        lms = (word_clm, char_clm, word_olm, char_olm, morph_olm)

        for pair in name_code_pairs:
            name = pair[0].replace(' ', '_') + '-' + pair[1]
            if pair in word_olm and pair in char_olm:
                lm_sizes[name] = len(word_olm[pair]) + len(char_olm[pair])

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
                    # if G or M occur without L, record globally
                    if 'G' in line.tag:
                        g_features(features, mention_dict, context, config)
                    if 'T' in line.tag:
                        t_features(features, mention_dict, context, config)
                    if 'M' in line.tag:
                        m_features(features, mention_dict, context, config)

            gl_features(features, mention_dict, context, config, common_table, eng_words, len(lang_names))
            w_features(features, mention_dict, context, config, len(lang_names), len(list(doc.lines())))
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
    sess = requests.Session()
    for row in reader:
        code = row[0]
        iso_code = row[8].strip()
        url = requests.compat.urljoin(baseuri, code + '.zip')
        combined_code = "{}_{}".format(iso_code, code)
        dest = os.path.join(output_dir, combined_code)

        logging.debug(
            'Downloading Crubadan data for {} from {}'.format(combined_code, url)
        )
        try:
            response = sess.get(url, timeout=30)
        except requests.exceptions.Timeout:
            logging.error(
                'Request timed out while trying to download data for {} from {}. Skipping...'
                .format(combined_code, url)
            )
            continue
        if response.status_code != requests.codes.ok:
            logging.error(
                'Failed to download the data for {} from {}. Response code {}. Skipping...'
                .format(combined_code, url, response.status_code)
            )
            continue
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
    """
    Build the LMs from the odin data
    """
    from lgid.buildlms import build_from_odin

    logging.info('Building ODIN LMs')
    indirec = config['locations']['odin-source']
    outdirec = config['locations']['odin-language-model']
    nc = 3
    nw = 2
    morph_split = config['parameters']['morpheme-delimiter']
    build_from_odin(indirec, outdirec, nc, nw, morph_split=morph_split)
    logging.info('Successfully built ODIN LMs')


if __name__ == '__main__':
    main()
