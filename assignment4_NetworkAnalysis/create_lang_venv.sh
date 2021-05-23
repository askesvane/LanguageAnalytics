#!/usr/bin/env bash

VENVNAME=network_env

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install jupyter

test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"
