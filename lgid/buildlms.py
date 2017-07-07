"""
Build language models from the ODIN data and monolingual data.
"""
#TODO: build from Crudaban
from xigt.codecs import xigtxml
from sklearn.feature_extraction.text import CountVectorizer
import re
import glob

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
        langcode = fname[13:].split('.')[0]
        xc = xigtxml.load(open(fname, 'r'))
        for igt in xc:
            langname = ""
            for met in igt.metadata:
                for oneMeta in met.metas:
                    if oneMeta.type == "language" and 'iso-639-3' in oneMeta.attributes:
                        langcode2 = oneMeta.attributes['iso-639-3']
                        if langcode == langcode2:
                            langname = oneMeta.attributes['name']
            for item in igt['c'].items:
                tag = item.attributes['tag']
                if re.match("^L(\+(CR|AL|DB|SEG))*$", tag):
                    if item.value():
                        if langname in texts:
                            texts[langname].append(item.value())
                        else:
                            texts[langname] = [item.value()]
        for name in texts:
            try:
                countsC = CountVectorizer(analyzer="char", ngram_range=(1, nc))
                countsC.fit_transform(texts[name])
                countsW = CountVectorizer(analyzer="word", ngram_range=(1, nw))
                countsW.fit_transform(texts[name])
                name = re.sub("/", "-", name)
                lmfileC = open(outdirec + "/" + langcode + "_" + name + ".char", "w")
                lmfileW = open(outdirec + "/" + langcode + "_" + name + ".word", "w")
                textC = ""
                for key in countsC.vocabulary_:
                    textC += key + "\t" + str(countsC.vocabulary_[key]) + '\n'
                lmfileC.write(textC)
                textW = ""
                for key in countsW.vocabulary_:
                    textW += key + "\t" + str(countsW.vocabulary_[key]) + '\n'
                lmfileW.write(textW)
            except ValueError:
                pass
