
"""
Feature functions for language identification

Category-level functions (gl_features(), w_features(), l_features(),
g_features(), and m_features()) call other functions for specific
features. Information extracted from documents to be used in
determining features comes from lgid.analyzers.

See README.md for more information about features.
"""

from collections import Counter
import numpy as np
import logging


from lgid.analyzers import (
    character_ngrams,
    word_ngrams
)

from lgid.util import (
    read_crubadan_language_model,
    read_odin_language_model
)

lm_dict = {}
percents = {}

def get_threshold_info():
    logging.info("LM threshold info:")
    for feat in percents:
        logging.info(feat)
        logging.info("\tMean: " + str(np.mean(percents[feat])))
        logging.info("\tStd. Dev: " + str(np.std(percents[feat])))


def gl_features(features, mentions, context, config, common_table):
    """
    Set matching global features to `True`

    Args:
        features: mapping from (lgname, lgcode) pair to features to values
        mentions: list of language mentions
        context: contextual information about the document
        config: model-building parameters
    """
    wsize = int(config['parameters']['window-size'])
    minfreq = int(config['parameters']['article-frequent-mention-threshold'])
    last = context['last-lineno']

    if config['features']['GL-first-lines']:
        window_mention('GL-first-lines', features, mentions, 0, wsize)

    if config['features']['GL-last-lines']:
        window_mention('GL-last-lines', features, mentions, last-wsize, last)

    if config['features']['GL-frequent']:
        frequent_mention('GL-frequent', features, mentions, minfreq, 0, last)

    if config['features']['GL-most-frequent']:
        frequent_mention('GL-most-frequent', features, mentions, None, 0, last)

    if config['features']['GL-most-frequent-code']:
        most_frequent_code(features, common_table)

    flag_common_words(features, config)


def w_features(features, mentions, context, config):
    """
    Set matching window features to `True`

    Args:
        features: mapping from (lgname, lgcode) pair to features to values
        mentions: list of language mentions
        context: contextual information about the document
        config: model-building parameters
    """
    wsize = int(config['parameters']['window-size'])
    a_wsize = int(config['parameters']['after-window-size'])
    c_wsize = int(config['parameters']['close-window-size'])
    ac_wsize = int(config['parameters']['after-close-window-size'])
    minfreq = int(config['parameters']['frequent-mention-threshold'])
    a_minfreq = int(config['parameters']['after-frequent-mention-threshold'])

    t = context['span-top']
    b = context['span-bottom']

    if config['features']['W-prev']:
        window_mention('W-prev', features, mentions, t-wsize, t)

    if config['features']['W-close']:
        window_mention('W-close', features, mentions, t-c_wsize, t)

    if config['features']['W-closest']:
        closest_mention('W-closest', features, mentions, t-wsize, t, t)

    if config['features']['W-frequent']:
        frequent_mention('W-frequent', features, mentions, minfreq, t-wsize, t)

    if config['features']['W-after']:
        window_mention('W-after', features, mentions, b, b+a_wsize)

    if config['features']['W-close-after']:
        window_mention('W-close-after', features, mentions, b, b+ac_wsize)

    if config['features']['W-closest-after']:
        closest_mention('W-closest-after', features, mentions, b, b+a_wsize, b)

    if config['features']['W-frequent-after']:
        frequent_mention('W-frequent-after', features, mentions, minfreq, b,
                         b+a_wsize)


def l_features(features, mentions, context, lms, config):
    """
    Set matching language (L) line features to `True`

    Args:
        features: mapping from (lgname, lgcode) pair to features to values
        mentions: list of language mentions
        context: contextual information about the document
        config: model-building parameters
    """
    line = context['line']

    if config['features']['L-in-line']:
        in_line_mention('L-in-line', features, mentions, line)

    word_clm, char_clm, word_olm, char_olm = lms
    pairs = list(features.keys())
    for name, code in pairs:
        # ODIN n-grams
        ngram_matching(features, 'L-LMw', line, name, code, False, 'odin', word_olm, config)
        ngram_matching(features, 'L-LMc', line, name, code, True, 'odin', char_olm, config)

        # Crubadan n-grams
        ngram_matching(features, 'L-CR-LMw', line, name, code, False, 'crubadan', word_clm, config)
        ngram_matching(features, 'L-CR-LMc', line, name, code, True, 'crubadan', char_clm, config)


def g_features(features, mentions, context, config):
    """
    Set matching gloss (G) line features to `True`

    Args:
        features: mapping from (lgname, lgcode) pair to features to values
        mentions: list of language mentions
        context: contextual information about the document
        config: model-building parameters
    """
    line = context['line']

    if config['features']['G-in-line']:
        in_line_mention('G-in-line', features, mentions, line)


def t_features(features, mentions, context, config):
    """
    Set matching translation (T) line features to `True`

    Args:
        features: mapping from (lgname, lgcode) pair to features to values
        mentions: list of language mentions
        context: contextual information about the document
        config: model-building parameters
    """
    line = context['line']

    if config['features']['T-in-line']:
        in_line_mention('T-in-line', features, mentions, line)


def m_features(features, mentions, context, config):
    """
    Set matching meta (M) line features to `True`

    Args:
        features: mapping from (lgname, lgcode) pair to features to values
        mentions: list of language mentions
        context: contextual information about the document
        config: model-building parameters
    """
    line = context['line']
    if config['features']['M-in-line']:
        in_line_mention('M-in-line', features, mentions, line)


def get_window(mentions, top, bottom):
    """
    Return language mentions that occur within a line-number window

    Args:
        mentions: list of language mentions
        top: top (i.e. smallest) line number in the window
        bottom: bottom (i.e. largest) line number in the window
    """
    for m in mentions:
        if m.startline <= bottom and m.endline >= top:
            yield m


def window_mention(feature, features, mentions, top, bottom):
    """
    Set *feature* to `True` for mentions that occur within the window

    Args:
        feature: feature name
        features: mapping from (lgname, lgcode) pair to features to values
        mentions: list of language mentions
        top: top (i.e. smallest) line number in the window
        bottom: bottom (i.e. largest) line number in the window
    """
    for m in get_window(mentions, top, bottom):
        features[(m.name, m.code)][feature] = True


def frequent_mention(feature, features, mentions, thresh, top, bottom):
    """
    Set *feature* to `True` for mentions that occur frequently

    Args:
        feature: feature name
        features: mapping from (lgname, lgcode) pair to features to values
        mentions: list of language mentions
        thresh: frequency threshold
        top: top (i.e. smallest) line number in the window
        bottom: bottom (i.e. largest) line number in the window
    """
    counts = Counter(
        (m.name, m.code) for m in get_window(mentions, top, bottom)
    )
    if thresh is None:
        if len(counts.values()):
            thresh = max(counts.values())
        else:
            thresh = 0
    for pair, count in counts.items():
        if count >= thresh:
            features[pair][feature] = True


def closest_mention(feature, features, mentions, top, bottom, ref):
    """
    Set *feature* to `True` for mentions that occur closest to *ref*

    Args:
        feature: feature name
        features: mapping from (lgname, lgcode) pair to features to values
        mentions: list of language mentions
        top: top (i.e. smallest) line number in the window
        bottom: bottom (i.e. largest) line number in the window
        ref: the reference line number for calculating distance
    """
    window = sorted(
        (abs(ref - m.startline), m)
        for m in get_window(mentions, top, bottom)
    )
    if window:
        smallest_delta = window[0][0]
        for delta, mention in window:
            if delta > smallest_delta:
                break
            features[(mention.name, mention.code)][feature] = True


def in_line_mention(feature, features, mentions, line):
    """
    Set *feature* to `True` for mentions that occur on *line*

    Args:
        feature: feature name
        features: mapping from (lgname, lgcode) pair to features to values
        mentions: list of language mentions
        line: FrekiLine object to inspect
    """
    for m in get_window(mentions, line.lineno, line.lineno):
        features[(m.name, m.code)][feature] = True

def ngram_matching(features, feature, line, name, code, characters, dataset, lm, config):
    if characters:
        threshold = float(config['parameters']['character-lm-threshold'])
        if dataset == 'odin':
            n = int(config['parameters']['character-n-gram-size'])
        elif dataset == 'crubadan':
            n = int(config['parameters']['crubadan-char-size'])
    else:
        threshold = float(config['parameters']['word-lm-threshold'])
        if dataset == 'odin':
            n = int(config['parameters']['word-n-gram-size'])
        elif dataset == 'crubadan':
            n = int(config['parameters']['crubadan-word-size'])

    if config['features'][feature]:
        if (name, code) in lm:
            lm = lm[(name, code)]
            ngrams = character_ngrams(line, (n, n)) if characters else word_ngrams(line, n)
            # remove the initial and final '\n' from Crubadan unigrams and all ODIN ngrams
            if dataset == 'odin' or n == 1:
                ngrams = ngrams[1:-1]

            matches = 0
            for ngram in ngrams:
                ngram = tuple(ngram)
                if ngram in lm:
                    matches += 1
            try:
                percent = matches / len(ngrams)
                if feature in percents:
                    percents[feature].append(percent)
                else:
                    percents[feature] = [percent]
            except ZeroDivisionError:
                return
            if percent >= threshold:
                features[(name, code)][feature] = True
            features[(name, code)][feature + '-percent'] = percent


def most_frequent_code(features, common_table):
    """
    Set feature to true if code is the most common one for language name
    :param features: mapping from (lgname, lgcode) pair to features to values
    :param common_table: language table mapping name to most common code
    :return: none
    """
    for name, code in features:
        if name in common_table:
            if code in common_table[name]:
                features[(name, code)]['GL-most-frequent-code'] = True


def flag_common_words(features, config):
    """
    Add features for languages that are often false positive language mentions
    :param features: mapping from (lgname, lgcode) pair to features to values
    :param config: config object
    :return: none
    """
    words = open(config['locations']['english-word-names'], 'r').read().split('\n')
    for name, code in features:
        if config['features']['GL-possible-english-word']:
            if name in words:
                features[(name, code)]['GL-possible-english-word'] = True
        if config['features']['GL-short-lang-name']:
            if len(name) <= int(config['parameters']['short-name-size']):
                features[(name, code)]['GL-short-lang-name'] = True

