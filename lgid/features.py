from freki.serialize import FrekiDoc
import re

from lgid.analyzers import (
    language_mentions
)
from lgid.util import (
    read_language_table,
    get_time,
)

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


def get_instances(infiles, config, labeled=True):
    """
    Find all instances of language text to be identified

    :param infiles: list of freki file paths
    :param config: config object
    :param labeled: True if processing data labeled with language names
    :return: texts: list of language strings,
            labels: list of gold labels for the texts,
            other_feats: list of dicts of mention features for each text
            ids: list of line_num, doc_id pairs
    """
    texts = []
    ids = []
    labels = []
    other_feats = []
    lgtable = read_language_table(config['locations']['language-table'])
    caps = config['parameters'].get('mention-capitalization', 'default')
    j = 1
    for file in infiles:
        print('Processing file ' + str(j) + '/' + str(len(infiles)))
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
        doc_id = int(re.search('doc_id=[0-9]+', all_text).group().split('=')[1])
        lines = all_text.split('\n')
        for i in range(len(lines)):
            line = lines[i]
            if "tag=L" in line:
                if labeled:
                    lang = re.search('lang_name=.+lang_code', line).group(0).split('=')[1].split('lang_code')[0].strip()
                    lang_code = re.search('lang_code=.+fonts', line).group(0).split('=')[1].split('fonts')[0].strip()
                    label = lang + '_' + lang_code
                    labels.append(label)
                text = line.split(':')[1]
                if type(text) == 'list':
                    text = ''.join(text)
                text = text.strip()
                text = re.sub('\s+', ' ', text)
                texts.append(text)
                feat_dict = get_features(i + 1, lg_dict, config)
                ids.append((i + 1, doc_id))
                feat_dict.update(glob_mentions)
                other_feats.append(feat_dict)
    return texts, labels, other_feats, ids