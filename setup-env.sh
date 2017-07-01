#!/bin/bash

BASE=.

if [ -d "$BASE/env" ]; then
    echo "Error: virtual environment at '$BASE/env/' already exists."
    exit 1
fi

echo "Creating virtual enviroment"
virtualenv -p python3 "$BASE/env"

echo "Entering virtual environment"
source "$BASE/env/bin/activate"

echo "Fetching dependencies"
pip install docopt scipy scikit-learn

TMP=`mktemp -d`
pushd "$TMP"
git clone https://github.com/xigt/freki.git
pip install ./freki
popd
rm -rf "$TMP"

echo "Exiting virtual environment"
deactivate
