#!/bin/sh
# Serve the site locally with the same files CI bundles into it.
set -e
cd "$(dirname "$0")"
rm -rf matrixgrad && cp -r ../matrixgrad matrixgrad
[ -f data/shakespeare.txt ] || curl -sSfL --create-dirs -o data/shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python3 -m http.server "${1:-8000}"
