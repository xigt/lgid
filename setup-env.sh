#!/bin/bash

BASE=.

if [ ! -d "$BASE/env" ]; then
    echo "Creating virtual enviroment"
    virtualenv -p python3 "$BASE/env"
fi

echo "Entering virtual environment"
source "$BASE/env/bin/activate"

echo "Fetching dependencies"
pip install docopt scipy scikit-learn requests

# Freki is not in PyPI, so install it from git if it isn't already installed
if ! $( python -c "import freki" 2>/dev/null >/dev/null ); then
    TMP=`mktemp -d`
    pushd "$TMP"
    git clone -b serialize-fixes https://github.com/xigt/freki.git
    pip install ./freki
    popd
    rm -rf "$TMP"
fi

echo "Exiting virtual environment"
deactivate
