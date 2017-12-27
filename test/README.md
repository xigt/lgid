## Testing

The `test/` subdirectory contains files for testing the `language_mentions` function
in `lgid/analyzers.py`. The `test/mentions_test.freki` file is the Freki file for
testing on. The `test/mentions_gold_output.txt` file is the gold standard file, and
the `test/mentions_single_gold_outut.txt` is the gold standard file for running
with the `single-mention` option turned on.

Run the command `./lgid.sh list-mentions config.ini test/mentions_test.freki` to
test against the `test/mentions_gold_output.txt` file. The output of the run should
match the file contents exactly.

Run the command `./lgid.sh list-mentions config.ini test/mentions_test.freki` with
the `single-longest-mention` setting in the config file equal to `yes` to test against
the `test/mentions_single_gold_output.txt` file. The output of the run should be
similar, but because the behavior of `single-longest-mention` is unspecified when
multiple mentions are the same length the output could be different without
failing the test.
