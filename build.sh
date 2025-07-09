#!/bin/bash
set -e

# 1. Проверка шрифтов
echo "--- Проверка шрифтов Arial ---"
mkdir -p fonts  # На всякий случай создаём папку

if [ -f "fonts/arial.ttf" ]; then
    echo "✅ Шрифты Arial найдены в папке fonts/"
    ls -l fonts/  # Вывод списка файлов для логов
else
    echo "❌ Ошибка: файл fonts/arial.ttf не найден!"
    echo "Убедитесь, что вы добавили в репозиторий:"
    echo "- fonts/arial.ttf"
    echo "- fonts/arialbd.ttf"
    echo "- fonts/ariali.ttf"
    echo "- fonts/arialbi.ttf"
    exit 1  # Прерываем сборку при ошибке
fi

# 2. Установка прав доступа (важно для Render.com)
chmod 644 fonts/*

# 3. Установка Python-зависимостей
echo "=== Установка Python-зависимостей ==="
python -m pip install --upgrade pip
pip install --cache-dir=.pip_cache -r requirements.txt

# 4. TextBlob (если нужно)
if grep -q "textblob" requirements.txt; then
    python -m textblob.download_corpora
fi

echo "=== Сборка успешно завершена ==="
