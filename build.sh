#!/bin/bash

# Install necessary Python packages for building the project
pip install setuptools wheel build

# Build the project using Python's build module
python -m build

# Install the built package from the generated distribution file
pip install ./dist/gh_explainer-0.0.0.tar.gz