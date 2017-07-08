"""
Build language models from the ODIN data and monolingual data.
"""

#TODO: build from Crudaban

from xigt.codecs import xigtxml
from sklearn.feature_extraction.text import CountVectorizer
from lgid.analyzers import  word_ngrams
import re
import glob
import numpy as np

def tokenizer(s):
    return s.split()

def build_from_odin(indirec, outdirec, nc, nw):
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
            countsC = CountVectorizer(analyzer="char", ngram_range=(1, int(nc)))
            cc = countsC.fit_transform([re.sub(" +", " ", texts[name])])
            countsW = CountVectorizer(analyzer="word", tokenizer=tokenizer, ngram_range=(1, int(nw)))
            cw = countsW.fit_transform([texts[name]])
            name = re.sub("/", "-", name)
            lmfileC = open(outdirec + "/" + langcode + "_" + name + ".char", "w")
            lmfileW = open(outdirec + "/" + langcode + "_" + name + ".word", "w")
            textC = ""
            for key in countsC.vocabulary_:
                count = cc[0, countsC.vocabulary_[key]]
                textC += key + "\t" + str(count) + '\n'
            lmfileC.write(textC)
            textW = ""
            for key in countsW.vocabulary_:
                count = cw[0, countsW.vocabulary_[key]]
                textW += key + "\t" + str(count) + '\n'
            lmfileW.write(textW)