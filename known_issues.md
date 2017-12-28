A small number of freki files are messed up in similar fashion. These files have space
characters (U+0020) in some or all of the contentful section of the file (meaning the
freki meta-information is uncorrupted) replaced with other, unusual whitespace characters.

The known problem characters are handled specially within the `list-mentions` function in
`lgid/analyzers.py`, but can potentially cause issues elsewhere. Splitting a string on
whitespace (`string.split()`) seems to work as expected and split on the bad characters
as well as on spaces, but any code that specifically expects or looks for space characters
will be thrown off by these files, which could result in crashes or unexpected output.

If you are writing code that expects or looks for space characters, it is recommended you
either replace any instances of the characters listed in the table below with spaces before
searching the text, or that you look for or allow for those characters in addition to spaces.

These characters can be produced or matched in Python strings by writing the Unicode character
code from the table below after `\u` within a string. For example, `'\u2028'` will result in a
string consisting only of the line separator character.

The following table lists the known files with this problem and which character the spaces
are replaced with in that file.

file                               | replacement character  | Unicode character code
---------------------------------- | ---------------------- | ---------------------
odin2.1-pdfs/2334.freki            | line separator         | 2028
odin2.1-pdfs/3525.freki            | paragraph separator    | 2029
