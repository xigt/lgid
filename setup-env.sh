#!/bin/bash

BASE=.

if [ ! -d "$BASE/env" ]; then
    echo "Creating virtual enviroment"
    virtualenv -p python3 "$BASE/env"
fi

echo "Entering virtual environment"
source "$BASE/env/bin/activate"

echo "Fetching dependencies"
pip install -U pip # make sure pip is fully up-to-date
pip install docopt~=0.6.2 scipy~=0.19 scikit-learn==0.19.1 requests~=2.18.4 unidecode~=0.4.21

# Freki is not in PyPI, so install it from git if it isn't already installed
if ! $( python -c "import freki" 2>/dev/null >/dev/null ); then
    TMP=`mktemp -d`
    pushd "$TMP"
    git clone --branch master https://github.com/xigt/freki.git
    pip install ./freki
    popd
    rm -rf "$TMP"
fi
# do the same for Xigt
if ! $( python -c "import xigt" 2>/dev/null >/dev/null ); then
    TMP=`mktemp -d`
    pushd "$TMP"
    git clone --branch master https://github.com/xigt/xigt.git
    pip install ./xigt
    popd
    rm -rf "$TMP"
fi

echo "Exiting virtual environment"
deactivate
