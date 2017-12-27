
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
import re


from lgid.analyzers import (
    character_ngrams,
    word_ngrams,
    morpheme_ngrams
)

from lgid.util import (
    read_crubadan_language_model,
    read_odin_language_model
)

lm_dict = {}


def gl_features(features, mentions, context, config, common_table, eng_words, num_langs):
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

    if config['features']['GL-first-lines'] == 'yes':
        window_mention('GL-first-lines', features, mentions, 0, wsize, num_langs, True)

    if config['features']['GL-last-lines'] == 'yes':
        window_mention('GL-last-lines', features, mentions, last-wsize, last, num_langs, True)

    if config['features']['GL-frequent'] == 'yes':
        frequent_mention('GL-frequent', features, mentions, minfreq, 0, last, num_langs, True)

    if config['features']['GL-most-frequent'] == 'yes':
        frequent_mention('GL-most-frequent', features, mentions, None, 0, last, num_langs, True)

    if config['features']['GL-most-frequent-code'] == 'yes':
        most_frequent_code(features, common_table, config)

    if config['features']['GL-is-english'] == 'yes' and ('english', 'eng') in features:
        features[('english', 'eng')]['GL-is-english'] = True

    flag_common_words(features, eng_words, config)


def w_features(features, mentions, context, config, num_langs, num_lines):
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
    if config['features']['W-prev'] == 'yes':
        window_mention('W-prev', features, mentions, t-wsize, t)

    if config['features']['W-close'] == 'yes':
        window_mention('W-close', features, mentions, t-c_wsize, t)

    if config['features']['W-closest'] == 'yes':
        closest_mention('W-closest', features, mentions, t-wsize, t, t)

    if config['features']['W-frequent'] == 'yes':
        frequent_mention('W-frequent', features, mentions, minfreq, t-wsize, t)

    if config['features']['W-after'] == 'yes':
        window_mention('W-after', features, mentions, b, b+a_wsize)

    if config['features']['W-close-after'] == 'yes':
        window_mention('W-close-after', features, mentions, b, b+ac_wsize)

    if config['features']['W-closest-after'] == 'yes':
        closest_mention('W-closest-after', features, mentions, b, b+a_wsize, b)

    if config['features']['W-frequent-after'] == 'yes':
        frequent_mention('W-frequent-after', features, mentions, minfreq, b,
                         b+a_wsize)
    if num_langs > 20:
        frequent_mention('W=500&langs>20-frequent', features, mentions, minfreq, t-500, t)
        frequent_mention('W=500&langs>20-frequent-after', features, mentions, minfreq, b,
                     b + 500)
    if num_lines > 2000:
        frequent_mention('W=500&lines>2000-frequent', features, mentions, minfreq, t - 500, t)
        frequent_mention('W=500&lines>2000-frequent-after', features, mentions, minfreq, b,
                         b + 500)


def l_features(features, mentions, context, lms, config):
    """
    Set matching language (L) line features to `True`

    Args:
        features: mapping from (lgname, lgcode) pair to features to values
        mentions: list of language mentions
        context: contextual information about the document
        lms: tuple of different language models
        config: model-building parameters
    """
    line = context['line']

    if config['features']['L-in-line'] == 'yes':
        in_line_mention('L-in-line', features, mentions, line)

    word_clm, char_clm, word_olm, char_olm, morph_olm = lms
    pairs = list(features.keys())
    # ODIN n-grams
    if config['features']['L-LMw'] == 'yes':
        ngram_matching(features, 'L-LMw', line, pairs, 'word', 'odin', word_olm, config)
    if config['features']['L-LMc'] == 'yes':
        ngram_matching(features, 'L-LMc', line, pairs, 'character', 'odin', char_olm, config)
    if config['features']['L-LMm'] == 'yes':
        ngram_matching(features, 'L-LMm', line, pairs, 'morpheme', 'odin', morph_olm, config)

    # Crubadan n-grams
    if config['features']['L-CR-LMw'] == 'yes':
        ngram_matching(features, 'L-CR-LMw', line, pairs, 'word', 'crubadan', word_clm, config)
    if config['features']['L-CR-LMc'] == 'yes':
        ngram_matching(features, 'L-CR-LMc', line, pairs, 'character', 'crubadan', char_clm, config)


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

    if config['features']['G-in-line'] == 'yes':
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

    if config['features']['T-in-line'] == 'yes':
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

    if config['features']['M-in-line'] == 'yes':
        in_line_mention('M-in-line', features, mentions, line)


def get_window(mentions, top, bottom):
    """
    Return language mentions that occur within a line-number window

    Args:
        mentions: list of language mentions
        top: top (i.e. smallest) line number in the window
        bottom: bottom (i.e. largest) line number in the window
    """
    for i in range(top, bottom + 1):
        if i in mentions:
            for m in mentions[i]:
                if m.startline <= bottom and m.endline >= top:
                    yield m

def add_nums(feature, features, pair, num_langs):
    for i in [5, 10, 15, 20, 40]:
        if num_langs < i:
            features[pair][feature + '&langs<' + str(i)] = True

def window_mention(feature, features, mentions, top, bottom, num_langs=0, add_num=False):
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
        if add_num:
            add_nums(feature, features, (m.name, m.code), num_langs)


def frequent_mention(feature, features, mentions, thresh, top, bottom, num_langs=0, add_num=False):
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
            if add_num:
                add_nums(feature, features, pair, num_langs)


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
        for m in get_window(mentions, top, bottom),
        #key=lambda x: (x[0], get_dist_to_ref(ref, x[1]))
    )
    if window:
        smallest_delta = window[0][0]
        for delta, mention in window:
            if delta > smallest_delta:
                break
            features[(mention.name, mention.code)][feature] = True


#def get_dist_to_ref(ref, m):
#    is_after = not bool(np.sign(ref - m.startline))
#    if is_after:
#        return -m.startcol
#    return m.endcol


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

def ngram_matching(features, feature, line, pairs, gram_type, dataset, lms, config):
    """
    Set *feature* to `True` for features involving n-gram matching

    Args:
        features: mapping from (lgname, lgcode) pair to features to values
        feature: feature name
        line: FrekiLine object to inspect
        pairs: list of (name, code) tuples
        gram_type: what type of gram to use: 'character', 'word', or 'morpheme'
        dataset: which language model to use: 'odin' or 'crubadan'
        lms: the language model object to use
        config: parameters
    """
    if gram_type == 'character':
        if dataset == 'odin':
            n = int(config['parameters']['character-n-gram-size'])
        elif dataset == 'crubadan':
            n = int(config['parameters']['crubadan-char-size'])
    elif gram_type == 'morpheme':
        n = int(config['parameters']['morpheme-n-gram-size'])
    else:
        if dataset == 'odin':
            n = int(config['parameters']['word-n-gram-size'])
        elif dataset == 'crubadan':
            n = int(config['parameters']['crubadan-word-size'])

    if gram_type == 'character':
        ngrams = character_ngrams(line, (n, n))
    elif gram_type == 'morpheme':
        ngrams = morpheme_ngrams(line, n, re.compile(config['parameters']['morpheme-delimiter']))
    else:
        ngrams = word_ngrams(line, n)

    # remove the initial and final '\n' from Crubadan unigrams and all ODIN ngrams except morpheme
    if (dataset == 'odin' or n == 1) and gram_type != 'morpheme':
        ngrams = ngrams[1:-1]

    if config['features'][feature]:
        for (name, code) in pairs:
            if (name, code) in lms:
                lm = lms[(name, code)]
                matches = 0
                for ngram in ngrams:
                    if ngram in lm:
                        matches += 1
                try:
                    percent = matches / len(ngrams)
                except ZeroDivisionError:
                    return
                inc = 0.1
                threshold = 0
                while threshold < 1:
                    threshold = round(threshold + inc, 2)
                    if percent >= threshold:
                        features[(name, code)][feature + '>' + str(threshold)] = True


def most_frequent_code(features, common_table, config):
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
        if len(name.split()) > 1 and config['features']['GL-multi-word-name'] == 'yes':
            features[(name, code)]['GL-multi-word-name'] = True


def flag_common_words(features, words, config):
    """
    Add features for languages that are often false positive language mentions
    :param features: mapping from (lgname, lgcode) pair to features to values
    :param config: config object
    :return: none
    """
    for name, code in features:
        if config['features']['GL-possible-english-word'] == 'yes':
            if name in words:
                features[(name, code)]['GL-possible-english-word'] = True
        if config['features']['GL-short-lang-name'] == 'yes':
            if len(name) <= int(config['parameters']['short-name-size']):
                features[(name, code)]['GL-short-lang-name'] = True
            for i in range(1, 10):
                if len(name) <= i:
                    features[(name, code)]['GL-C-name<' + str(i)] = True


def get_mention_by_lines(mentions):
    mention_dict = {}
    for m in mentions:
        key = m.startline
        if key not in mention_dict:
            mention_dict[key] = []
        mention_dict[key].append(m)
    return mention_dict
