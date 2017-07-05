#!/bin/bash

curdir=$( cd `dirname $0` && pwd)

if [ ! -d "$curdir/env" ]; then
    echo "Virtual environment does not exist; run the following then try again:"
    echo "    setup-env.sh"
    exit 1
fi

source "$curdir/env/bin/activate"
python -m lgid.main "$@"
deactivate
