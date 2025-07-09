#!/bin/bash
set -e

echo "=== Installing dependencies ==="
python -m pip install --upgrade pip
pip install -r requirements.txt

if grep -q "textblob" requirements.txt; then
    echo "=== Installing TextBlob corpora ==="
    python -m textblob.download_corpora
fi
