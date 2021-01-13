#!/bin/bash -e

echo "Installing dependencies"

python -m pip install --upgrade pip build
python -m pip install --upgrade -r requirements.txt

if [ -n "${OPTIONAL_DEPENDS}" ]; then
    for DEP in ${OPTIONAL_DEPENDS}; do
        pip install $DEP || true
    done
fi

echo "Finished installed dependencies"
