#!/bin/bash
set -e

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt

# Установка данных для TextBlob (только если textblob в requirements)
if grep -q "textblob" requirements.txt; then
    python -m textblob.download_corpora
fi
