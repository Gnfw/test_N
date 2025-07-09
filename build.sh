set -e

echo "--- Installing dependencies ---"
pip install --upgrade pip
pip install -r requirements.txt

if [ -f "requirements.txt" ] && grep -q "textblob" requirements.txt; then
    echo "--- Installing TextBlob corpora ---"
    python -m textblob.download_corpora
fi
