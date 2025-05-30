#!/bin/bash

# create a new virtual environment
python3 -m venv env/

# activate it
source env/bin/activate

# install SynaptogenML locally end editable with all dependencies
pip install -e .[examples,tests]

# to start the interactive python notebooks call:
# jupyter notebook
