"""
Build language models from the ODIN data and monolingual data.
"""


from xigt.codecs import xigtxml
from sklearn.feature_extraction.text import CountVectorizer
import re
import os
import glob
from lgid import analyzers
from lgid.util import hard_normalize_characters


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
    Builds character, morpheme and word language models from a directory of ODIN xml files, with
    each language in a separate file named with the language code.

    The LMs are output in the format ngram \t count \n. The filenames have the format
    languageCode_languageName.char/.word/.morph.

    Parameters:
        indirec: the directory containing the odin data
        outdirec: the lm output directory
        nc: maximum ngram length for characters
        nw: maximum ngram length for words
        lhs: left-padding character (to show token boundaries)
        rhs: right-padding character (to show token boundaries)
        morph_split: string containing morpheme delimiting characters
    """
    if not os.path.exists(indirec):
        exit("The 'odin-source' directory specified in the config file does not exist. Quitting.")
    if not os.path.exists(outdirec):
        os.makedirs(outdirec)
    for fname in glob.glob(os.path.join(indirec, "*.xml")):
        texts = {}
        langcode = re.search('(.{3})\.xml', fname).group(1)
        langcode = re.sub("/", "-", langcode)
        langcode = hard_normalize_characters(langcode)
        xc = xigtxml.load(open(fname, 'r', encoding='utf8'))
        for igt in xc:
            langname = ""
            for met in igt.metadata:
                for oneMeta in met.metas:
                    subjectChild = None
                    for child in oneMeta.children:
                        if child.name == 'subject':
                            subjectChild = child
                            break
                    if subjectChild != None and '{http://www.language-archives.org/OLAC/1.1/}code' in subjectChild.attributes:
                        langcode2 = subjectChild.attributes['{http://www.language-archives.org/OLAC/1.1/}code']
                        if langcode == langcode2:
                            langname = subjectChild.text.lower()
                            langname = hard_normalize_characters(langname)
                            langname = re.sub("/", "-", langname)
                            break
            if langname:
                for tier in [x for x in igt if x.type == 'odin' and x.attributes['state'] == 'normalized']:
                    for item in tier.items:
                        tag = item.attributes['tag']
                        if re.match("^L(\+(CR|AL|DB|SEG))*$", tag):
                            if item.value():
                                key = langcode + "_" + langname
                                if key in texts:
                                    texts[key] += " " + item.value()
                                else:
                                    texts[key] = item.value()
                                if langcode in texts:
                                    texts[langcode] += " " + item.value()
                                else:
                                    texts[langcode] = item.value()
        for name in texts:
            source = re.sub(" +", " ", texts[name])
            countsC = CountVectorizer(analyzer=lambda doc: analyzers.character_ngrams(doc, (1, int(nc)), lhs, rhs))
            cc = countsC.fit_transform([source])
            countsW = CountVectorizer(analyzer="word", tokenizer=tokenizer, ngram_range=(1, int(nw)))
            cw = countsW.fit_transform([source])
            morph = morph_tokenizer(morph_split)
            countsM = CountVectorizer(analyzer="word", tokenizer=morph.tok, ngram_range=(1, int(nw)))
            cm = countsM.fit_transform([source])
            print(name)
            lmfileC = open(os.path.join(outdirec, name + ".char"), "w", encoding='utf8')
            lmfileW = open(os.path.join(outdirec, name + ".word"), "w", encoding='utf8')
            lmfileM = open(os.path.join(outdirec, name + ".morph"), "w", encoding='utf8')

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

            lmfileC.close()
            lmfileW.close()
            lmfileM.close()
