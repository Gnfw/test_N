#!/bin/bash
set -e

# Принудительно устанавливаем Python 3.10
pyenv install 3.10.12 -s
pyenv global 3.10.12

# Проверяем версию
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt

# Установка данных TextBlob (если есть в requirements)
if grep -q "textblob" requirements.txt; then
    python -m textblob.download_corpora
fi
