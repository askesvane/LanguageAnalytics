#!/usr/bin/env bash

VENVNAME=DeepLearning_env

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install jupyter

test -f requirements.txt && pip install -r requirements.txt
python -m spacy download en_core_web_sm

deactivate
echo "build $VENVNAME"
