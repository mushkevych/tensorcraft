#!/usr/bin/env bash

set -xeu -o pipefail

# Install spacy's en_core_web_sm component
python -m spacy download en_core_web_sm
