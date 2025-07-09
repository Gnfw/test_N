#!/bin/bash
set -e

# Установка зависимостей с предпочтением бинарных дистрибутивов
pip install --upgrade pip
pip install --prefer-binary -r requirements.txt

# Установка данных для TextBlob
python -m textblob.download_corpora