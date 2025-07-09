#!/bin/bash
set -e

# Установка шрифтов Arial
echo "--- Installing Arial fonts ---"
mkdir -p fonts

# Скачиваем Arial с Google Fonts
if [ ! -f "fonts/arial.ttf" ]; then
    echo "Скачиваем шрифты Arial..."
    
    # Скачиваем архив с Arial
    wget -q -O arial.zip "https://fonts.google.com/download?family=Arial"
    
    # Распаковываем нужные файлы
    unzip -j arial.zip "**/Arial-Regular.ttf" -d fonts/
    unzip -j arial.zip "**/Arial-Bold.ttf" -d fonts/
    unzip -j arial.zip "**/Arial-Italic.ttf" -d fonts/
    unzip -j arial.zip "**/Arial-BoldItalic.ttf" -d fonts/
    
    # Переименовываем файлы в нужный формат
    mv fonts/Arial-Regular.ttf fonts/arial.ttf
    mv fonts/Arial-Bold.ttf fonts/arialbd.ttf
    mv fonts/Arial-Italic.ttf fonts/ariali.ttf
    mv fonts/Arial-BoldItalic.ttf fonts/arialbi.ttf
    
    # Удаляем временные файлы
    rm -f arial.zip
    echo "Шрифты Arial успешно установлены!"
fi

echo "=== Installing dependencies ==="
python -m pip install --upgrade pip
pip install -r requirements.txt

if grep -q "textblob" requirements.txt; then
    echo "=== Installing TextBlob corpora ==="
    python -m textblob.download_corpora
fi
