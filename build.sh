#!/bin/bash

# Mudhumeni AI Build Script for Render Deployment
echo "🚀 Starting Mudhumeni AI build process..."

# Update pip to latest version
echo "📦 Updating pip..."
pip install --upgrade pip

# Install setuptools and wheel first
echo "🔧 Installing build tools..."
pip install --upgrade setuptools wheel

# Install requirements
echo "📋 Installing dependencies..."
pip install -r requirements.txt

# Install NLTK data
echo "📚 Installing NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

echo "✅ Build completed successfully!"
