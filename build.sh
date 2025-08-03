#!/bin/bash

# Simple build script for Render deployment
echo "🚀 Building Mudhumeni AI..."

# Install requirements
pip install -r requirements.txt

# Install NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('vader_lexicon', quiet=True)"

echo "✅ Build completed!"
