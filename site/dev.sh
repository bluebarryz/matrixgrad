#!/bin/sh
# Builds the site and serves it locally.
set -e
cd "$(dirname "$0")"
./build.sh
python3 -m http.server "${1:-8000}"
