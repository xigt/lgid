"""
Build language models from the ODIN data and monolingual data.
"""


from xigt.codecs import xigtxml
from sklearn.feature_extraction.text import CountVectorizer
from lgid.analyzers import word_ngrams, character_ngrams
import re
import glob
import numpy as np
from lgid import analyzers
from util import normalize_characters


def tokenizer(s):
    return s.split()


class morph_tokenizer:
    """
    Tokenizes strings by morpheme
    """
    def __init__(self, split):
        """
        :param split: string containing morpheme delimiting characters
        """
        self.split = split

    def tok(self, s):
        """
        split a string into morphemes
        :param s: string
        :return: list of morpheme strings
        """
        return re.split(self.split, s)


def build_from_odin(indirec, outdirec, nc, nw, lhs='<', rhs='>', morph_split='[\s\-\=\+]+'):
    """
    Builds character and word language models from a directory of ODIN xml files, with each language in a separate
    file named with the language code.

    The LMs are output in the format ngram \t count \n. The filenames have the format
    languageCode_languageName.char/.word.

    Parameters:
        indirec: the directory containing the odin data
        outdirec: the lm output directory
        nc: maximum ngram length for characters
        nw: maximum ngram length for words
        lhs: left-padding character (to show token boundaries)
        rhs: right-padding character (to show token boundaries)
        morph_split: string containing morpheme delimiting characters
    """
    for fname in glob.glob(indirec + "/*.xml"):
        texts = {}
        langcode = re.search('.{3}\.xml', fname).group(0).split('.')[0]
        xc = xigtxml.load(open(fname, 'r'))
        for igt in xc:
            langname = ""
            for met in igt.metadata:
                for oneMeta in met.metas:
                    if oneMeta.type == "language" and 'iso-639-3' in oneMeta.attributes:
                        langcode2 = oneMeta.attributes['iso-639-3']
                        if langcode == langcode2:
                            langname = oneMeta.attributes['name']
            if langname:
                for item in igt['c'].items:
                    tag = item.attributes['tag']
                    if re.match("^L(\+(CR|AL|DB|SEG))*$", tag):
                        if item.value():
                            if langname in texts:
                                texts[langname] += " " + item.value()
                            else:
                                texts[langname] = item.value()
        for name in texts:
            source = re.sub(" +", " ", texts[name])
            countsC = CountVectorizer(analyzer=lambda doc: analyzers.character_ngrams(doc, (1, int(nc)), lhs, rhs))
            cc = countsC.fit_transform([source])
            countsW = CountVectorizer(analyzer="word", tokenizer=tokenizer, ngram_range=(1, int(nw)))
            cw = countsW.fit_transform([source])
            morph = morph_tokenizer(morph_split)
            countsM = CountVectorizer(analyzer="word", tokenizer=morph.tok, ngram_range=(1, int(nw)))
            cm = countsM.fit_transform([source])
            name = re.sub("/", "-", name)
            norm_name = normalize_characters(name)
            lmfileC = open(outdirec + "/" + langcode + "_" + norm_name + ".char", "w")
            lmfileW = open(outdirec + "/" + langcode + "_" + norm_name + ".word", "w")
            lmfileM = open(outdirec + "/" + langcode + "_" + norm_name + ".morph", "w")
            textC = ""
            for key in countsC.vocabulary_:
                count = cc[0, countsC.vocabulary_[key]]
                textC += ''.join(key) + "\t" + str(count) + '\n'
            lmfileC.write(textC)
            textW = ""
            for key in countsW.vocabulary_:
                count = cw[0, countsW.vocabulary_[key]]
                textW += key + "\t" + str(count) + '\n'
            lmfileW.write(textW)
            textM = ""
            for key in countsM.vocabulary_:
                count = cm[0, countsM.vocabulary_[key]]
                textM += key + "\t" + str(count) + '\n'
            lmfileM.write(textM)
