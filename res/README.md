
# Resource files

Note: I inherited the precursor to this project, and I don't know how
the original data files were obtained, so I am including this notice in
case there are any licensing issues. I checked in the `lang_table.txt`
and `Crubadan.csv` files as I could not find them readily available
online, and I don't see any indication that such inclusion would be
improper use. If any of these files should be removed from this
repository, please file an [issue](https://github.com/xigt/lgid/issues).

## ODIN

The ODIN data used is in the odin-by-lang directory. Each file contains IGT for one language
identified by its code, and is in the xigt format. 

The odin_lm directory contains word and char LMs built from the odin data.

## Crúbadán

The [Crúbadán][] data is under the [CC BY 4.0][] license, and that
possibly extends to the included `res/Crubadan.csv` file, although I
don't see the CSV file available on the Crúbadán website.

## lang_table

The `res/lang_table.txt` file appears to be generated from the [ODIN][]
data, but it could be produced from something else.

## common_codes

The `res/common_codes.txt` is generated from the lang_table. It pairs a language name with the code most commonly associated with it. This file was trained on the ODIN data using the find-common-codes command.

## english_word_language_names

The `res/english_word_language_names.txt` contains language names that are also ENglish words or proper names, and so are often false positive language mentions predicted by the system.

[Crúbadán]: http://crubadan.org/
[CC BY 4.0]: https://creativecommons.org/licenses/by/4.0/
[ODIN]: http://depts.washington.edu/uwcl/odin/

