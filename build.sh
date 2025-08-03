# Build script for Render
# This file will be executed during deployment

echo "Installing NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

echo "Build completed successfully!"
