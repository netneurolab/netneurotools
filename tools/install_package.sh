#!/bin/bash -e

echo "Building and installing netneurotools"

python -m build

if [ "$INSTALL_TYPE" == "setup" ]; then
  python -m pip install .
elif [ "$INSTALL_TYPE" == "sdist" ]; then
  python -m pip install dist/*.tar.gz
elif [ "$INSTALL_TYPE" == "wheel" ]; then
  python -m pip install dist/*.whl
else
  false
fi

pip install "netneurotools[$CHECK_TYPE]"

echo "Done installing netneurotools"
