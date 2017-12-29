# lgid

Language identification of linguistic examples

This utility identifies the subject language of linguistic examples,
specifically for use in the [ODIN][] data-acquisition pipeline. Unlike
common language identification methods that look for characteristic
n-grams of characters or words, this tool attempts to identify languages
for which there might be little or no other example data. This is
accomplished by looking for language mentions in the context of the
examples. If a language name is mentioned in the document containing the
example, it is a candidate for the subject language of the example.
Other features, such as the proximity of language mentions to the
example, or n-gram matches to a language model (if such a model exists)
also contribute to the determination.

In addition to accounting for a lack of language model data, this
package also does not assume any supervised training data. In order to
make a classifier that can predict a language when no positive instances
of the language have been seen, we generalize the problem so the
classifier predicts the probability that some mentioned language is
associated with the current example. The language with the highest
probability is then chosen.

The `lgid` package includes functionality for building resources from
training data, listing language mentions in a document, training models,
classifying data, and testing the trained models.

## Installation

The basic requirements are probably satisfied by any modern Linux or Mac setup:
- bash
- git
- Python (3.3 or higher)
- [virtualenv][]

First clone the repository:

```bash
~$ git clone https://github.com/xigt/lgid.git
```

After cloning this repository, run the `setup-env.sh` script to create a
[virtual environment][virtualenv] and download the necessary
dependencies into it:

```bash
~$ cd lgid/
~/lgid$ bash setup-env.sh
```

The dependencies installed by this script are:
- [docopt][] (0.6)
- [SciPy][] (0.19)
- [scikit-learn][] (0.19)
- [Requests][] (2.18)
- [Unidecode][] (0.4)
- [Freki][]
- [Xigt][]

When completed, run `lgid.sh` as the front end to all tasks, as it
manages the activation/deactivation of the virtual environment:

```bash
~/lgid$ ./lgid.sh 
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

```

Try `lgid.sh --help` for more usage information.

## Examples of Usage

Test the performance of the prebuilt model on the included sample input:

```bash
./lgid.sh -v test --model=model/sample_model config.ini sample/input/*
```

Classify the sample input and output the files with predicted languages marked:

```bash
./lgid.sh -v classify --model=model/sample_model --out=sample/output config.ini sample/input/*
```

## Code and Resources

Here is an overview of the imporant files in the repository:

```bash
lgid
├── setup-env.sh        # first-time setup
├── lgid.sh             # main script for running the program
├── config.ini          # configuration parameters
├── lgid                # source code
│   ├── analyzers.py    # functions for extracting info from input documents
│   ├── buildlms.py     # functions for building language model files out of ODIN data
│   ├── features.py     # functions for activating model features
│   ├── main.py         # main process; entry point for the lgid.sh command
│   ├── models.py       # abstraction of the machine learning model(s)
│   └── util.py         # utility functions for reading/transforming resources
├── res                 # resources for running the program
│    ├── Crubadan.csv    # index file for downloading Crúbadán data
│    ├── lang_table.txt  # language name-code mapping
│    ├── common_codes.txt  # Shows code most commonly paired with each language name
│    ├── english_word_language_names.txt # list of language names that are English words
│    ├── crubadan_directory_index.csv # table of what directory holds Crúbadán data for each language
│    ├── language_index.txt # table of each language and its ID
│    ├── word_index.txt # table of each word present in a language name and its ID
│    ├── word_language_mapping.txt # table of each word present in a language name and which languages it appears in (in ID form)
│    └── odin-lm.zip    # files for building the ODIN language model
├── sample              # results from sample runs
└── test                # files for testing the program
     ├── mentions_gold_output.txt   # the gold standard output when running list-mentions on mentions_test.freki
     ├── mentions_single_gold_output.txt   # the gold standard output when running list-mentions on mentions_test.freki with the setting `single-longest-mention = yes`
     └── mentions_test.freki  # freki file for testing list-mentions
```

In the repository, the `lgid/` subdirectory contains all code for data
analysis, model building, and document classification.
The `res/` subdirectory contains resource files used in model-building.
Only static files that we have rights to, like the language name-code
mapping table, should be checked in. Other resources, like the compiled
language model or [Crúbadán][] data, may reside here on a local machine,
but they should not be committed to the remote repository.

## File Formats

All of the functions that take `INFILE` as an argument expect that file or files to be in [Freki](https://github.com/xigt/freki) format.
The `classify` function produces Freki files as output.

The `build-odin-lm` function expects its input files (location specified in the config file) to be in the [Xigt][] format.

The [ODIN][] language model files have one ngram on each line, with the format `<ngram>\t<count>`. There are no special symbols
used for beginning or end of line. Each file contains ngrams for all values of n, 1-3 for characters and 1-2 for words. The
morpheme language models are built using the word data.

The [Crúbadán][] language model files have one ngram on each line, with the format `<ngram> <count>`. The `\n` character is used
to indicate beginning or end of line for word ngrams. The `<` and `>` characters are used for the beginning and end of word, respectively,
for character ngrams. Each file contains ngrams for only one value of n. The Crúbadán language models have only
trigrams for characters and both unigrams and bigrams for words.

## Configuration

The `config.ini` file contains parameters for managing builds of the
model.

The `[locations]` section contains paths for finding various
resources.

location                           | description
---------------------------------- | -----------
lgid-dir                           | location of the system files on disk
language-table                     | location of the master language table
most-common-codes                  | location of table of most common codes for each language
english-word-names                 | location of list of languages whose names are also English words
word-index                         | location of the file mapping words to IDs
language-index                     | location of the file mapping languages to IDs
word-language-mapping              | location of the file mapping words to the languages they appear in
odin-source                        | location of the Xigt files for building ODIN language models from
odin-language-model                | directory containing the ODIN language model files
crubadan-index                     | location of file containing index and download location for Crúbadán language model data
crubadan-base-uri                  | base URL where Crúbadán data files are downloaded from
crubadan-language-model            | directory containing the Crúbadán language model
crubadan-directory-index           | location of table tracking location of Crúbadán language model for each language
classify-error-file | language id errors are written to this text file

The `[parameters]` section contains parameters for modifying
the behavior of feature functions. Available parameters are described
below:

parameter                          | description
---------------------------------- | -----------
window-size                        | number of lines before an IGT to consider
after-window-size                  | number of lines after an IGT to consider
close-window-size                  | smaller window before an IGT
after-close-window-size            | smaller window after an IGT
word-n-gram-size                   | number of tokens in word-lm n-grams
morpheme-n-gram-size               | number of tokens in morpheme-lm n-grams
character-n-gram-size              | number of chars in character-lm n-grams
crubadan-char-size                 | number of chars in Crúbadán character-lm n-grams
crubadan-word-size                 | number of tokens in Crúbadán word-lm n-grams
morpheme-delimiter                 | regular expression for tokenizing morphemes
frequent-mention-threshold         | minimum window mentions to be "frequent"
after-frequent-mention-threshold   | min. mentions after an IGT to be "frequent"
article-frequent-mention-threshold | min. mentions in document to be "frequent"
mention-capitalization             | case-normalization for language mentions
short-name-size             | a language name shorter than or equal to this length is flagged, as very short names are often false positive mentions
single-longest-mention             | return all mentions in a given span or only the single longest one. `yes` for on, anything else for off
code-only-odin-lms                 | use ODIN language models for code+language or just the code. `yes` for on, anything else for off


The `[features]` section has boolean flags for turning on/off specific
features. The value `yes` turns the feature on, any other value turns it
off. The available features are:

feature name     | description    | notes
---------------- | -------------- | -------
GL-first-lines   | language mentioned in the first window of the document
GL-last-lines    | language mentioned in the last window of the document
GL-frequent      | language mentioned `N+` times in the document
GL-most-frequent | language is the most frequently mentioned one
GL-most-frequent-code | code is the most frequent one paired with language
GL-possible-english-word | language name is possibly an English word or name
GL-short-lang-name | language name is shorter than short-name-size, and may be a false positive because it occurs as a word in some language
GL-is-english | language is English
GL-multi-word-name | language name is multiple words
W-prev           | language mentioned within the IGT's preceding window
W-close          | language mentioned within a smaller preceding window
W-closest        | language is closest to the IGT in the preceding window
W-frequent       | language mentioned `N+` times in the preceding window
W-after          | language mentioned within the IGT's following window
W-close-after    | language mentioned within a smaller following window
W-closest-after  | language is closest to the IGT in the following window
W-frequent-after | language mentioned `N+` in the following window
L-in-line        | language mentioned in the IGT's language line
G-in-line        | language mentioned in the IGT's gloss line
T-in-line        | language mentioned in the IGT's translation line
M-in-line        | language mentioned in the IGT's meta lines
L-LMw            | more than `M%` of word ngrams occur in the training data, using ODIN data
L-LMm            | more than `M%` of morpheme ngrams occur in training data, using ODIN data
L-LMc            | more than `M%` of character ngrams occur in training data, using ODIN data
L-CR-LMw         | same as L-LMw, but using Crúbadán data
L-CR-LMc         | same as L-LMc, but using Crúbadán data
G-overlap        | at least `M%` of gloss tokens occur in the training data | not implemented
W-prevclass      | language is predicted for the previous IGT | not implemented

Note that the features have prefixes that group them into categories.
The categories are:

feature prefix | description
-------------- | -----------
GL-            | feature is relevant globally
W-             | feature is relevant within a window
L-             | feature is relevant for the language line
G-             | feature is relevant for the gloss line
T-             | feature is relevant for the translation line
M-             | feature is relevant for a meta line


[virtualenv]: https://virtualenv.pypa.io/
[ODIN]: http://depts.washington.edu/uwcl/odin/
[Crúbadán]: http://crubadan.org/
[docopt]: http://docopt.org/
[SciPy]: https://www.scipy.org/
[scikit-learn]: http://scikit-learn.org
[Requests]: http://docs.python-requests.org/
[Unidecode]: https://pypi.python.org/pypi/Unidecode/
[Freki]: https://github.com/xigt/freki
[Xigt]: http://depts.washington.edu/uwcl/xigt
