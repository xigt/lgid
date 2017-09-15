# Change Log

## [Unreleased][unreleased]

This initial version ports functionality from the original language
identifier written, which was not publicly released. This version is
similar in capabilities, but cleans up the code to make it more
maintainable and easy to extend.

### Added

- Building char, word, and morpheme LMs from ODIN
- L-LM features
- L-LM-CR features
- Features for dealing with language mention false positives: GL-short-lang-name, GL-possible-english-word
- GL-most-frequent-code feature
- GL-is-english feature
- G-in-line feature
- T-in-line feature

### Removed

- LM threshold in config removed in favor of creating several features with different thresholds. This dramatically increases accuracy.

### Fixed

- Dict shallow copying bug caused features for languages to persist across instances in a single file.
- Bug in util.py:spans(doc) caused the first line of a span to be considered a separate span.
- Optimized searching through mentions to only consider mentions in necessary window.
- Increased LM overlap computation speed by making LMs sets rather than lists.

### Changed

- Each LM feature now corresponds to ten features with thresholds between 0 and 1 in increments of 0.1.

### Deprecated

- W-prevclass would require reworking system structure

[unreleased]: ../../tree/develop
[v0.1.0]: ../../releases/tag/v0.1.0
[README]: README.md
