#!/bin/bash
set -e

# Установка шрифтов
echo "--- Installing DejaVu fonts ---"
mkdir -p fonts
if [ ! -f "fonts/DejaVuSans.ttf" ]; then
    wget -q -O fonts/dejavu.zip https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip
    unzip -q fonts/dejavu.zip -d fonts_temp
    cp fonts_temp/dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf fonts/
    rm -rf fonts_temp dejavu.zip
fi

echo "=== Installing dependencies ==="
python -m pip install --upgrade pip
pip install -r requirements.txt

if grep -q "textblob" requirements.txt; then
    echo "=== Installing TextBlob corpora ==="
    python -m textblob.download_corpora
fi
