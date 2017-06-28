# lgid
language identification of linguistic examples

## Model Parameters

parameter              | description
---------------------- | -----------
window-size            | number of lines before an IGT to consider
after-window-size      | number of lines after an IGT to consider
mention-threshold      | number of mentions to be considered "frequent"
word-lm-threshold      | percent of overlap required for word-lm features
morpheme-lm-threshold  | percent of overlap required for morpheme-lm features
character-lm-threshold | percent of overlap required for character-lm features
gloss-lm-threshold     | percent of overlap required for gloss-lm features

## Language ID Model Features

feature prefix | description
-------------- | -----------
GL-            | feature is relevant globally
W-             | feature is relevant within a window
L-             | feature is relevant for the language line
G-             | feature is relevant for the gloss line
M-             | feature is relevant for a meta line

name             | description
---------------- | -----------
GL-first-lines   | language mentioned in the first window of the document
GL-last-lines    | language mentioned in the last window of the document
GL-frequent      | language mentioned `N+` times in the document
GL-most-frequent | language is the most frequently mentioned one
W-prev           | language mentioned within the IGT's preceding window
W-close          | language mentioned within a smaller preceding window
W-closest        | language is closest to the IGT in the preceding window
W-frequent       | language mentioned `N+` times in the preceding window
W-after          | language mentioned within the IGT's following window
W-close-after    | language mentioned within a smaller following window
W-closest-after  | language is closest to the IGT in the following window
W-frequent-after | language mentioned `N+` in the following window
L-in-line        | language mentioned in the IGT's language line
M-in-line        | language mentioned in the IGT's meta lines
L-LMw            | at least `M%` of word ngrams occur in the training data
L-LMm            | at least `M%` of morpheme ngrams occur in training data
L-LMc            | at least `M%` of character ngrams occur in training data
G-overlap        | at least `M%` of gloss tokens occur in the training data
L-CR-LMw         | same as L-LMw, but for Crubadan data
L-CR-LMm         | same as L-LMm, but for Crubadan data
L-CR-LMc         | same as L-LMc, but for Crubadan data
W-prevclass      | language is predicted for the previous IGT
