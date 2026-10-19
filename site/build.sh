#!/bin/sh
# Assembles the static site: the engine it runs, the text it trains on, and the Recorded Trace.
set -e
cd "$(dirname "$0")"
rm -rf matrixgrad && cp -r ../matrixgrad matrixgrad && rm -rf matrixgrad/__pycache__
[ -f data/shakespeare.txt ] || curl -sSfL --create-dirs -o data/shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
[ -f recorded/run.json ] || PYTHONPATH=..:py python3 py/record_trace.py data/shakespeare.txt recorded
