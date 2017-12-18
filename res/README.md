
# Resource files

Note: I inherited the precursor to this project, and I don't know how
the original data files were obtained, so I am including this notice in
case there are any licensing issues. I checked in the `lang_table.txt`
and `Crubadan.csv` files as I could not find them readily available
online, and I don't see any indication that such inclusion would be
improper use. If any of these files should be removed from this
repository, please file an [issue](https://github.com/xigt/lgid/issues).

## Prebuilt Language Models

The zip file odin-lm.zip contains word, character, and morpheme language models for 1999 languages. If you want to use your own LMs or build LMs from your own files, you can change the locations in `config.ini`, or download the raw ODIN data to the location in the config. The LM format is described in the parent `README.md` file under the
`File Formats` section

## ODIN

The [ODIN][] data used is in the odin-by-lang directory. Each file contains IGT for one language
identified by its code, and is in the xigt format. 

## Crúbadán

The [Crúbadán][] data is under the [CC BY 4.0][] license, and that
possibly extends to the included `res/Crubadan.csv` file.
The `res/Crubadan.csv` file is the CSV download available on the Crúbadán website.
This one is built from multiple downloads (one with the website sorted A-Z, one with
the website sorted Z-A), as the Crúbadán website caps the file at 2000 rows.

## lang_table

The `res/lang_table.txt` file appears to be generated from the [ODIN][]
data, but it could be produced from something else.

## common_codes

The `res/common_codes.txt` is generated from `lang_table.txt`. It pairs a language name with the code most commonly associated with it. This file was trained on the ODIN data using the find-common-codes command.

## english_word_language_names

The `res/english_word_language_names.txt` file contains language names that are also English words or proper names, and so are often false positive language mentions predicted by the system. This list is hand-curated.

## crubadan_directory_index

The `res/crubadan_directory_index.csv` file contains information on which subdirectory of Crúbadán data to use for each language. It built by a script
and then manually edited and cleaned.

[Crúbadán]: http://crubadan.org/
[CC BY 4.0]: https://creativecommons.org/licenses/by/4.0/
[ODIN]: http://depts.washington.edu/uwcl/odin/

