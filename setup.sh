#!/bin/bash

# Create virtual environment
python -m venv venv

# Activate virtual environment
if [ "$(uname)" == "Darwin" ] || [ "$(uname)" == "Linux" ]; then
    source venv/bin/activate
elif [ "$(expr substr $(uname -s) 1 5)" == "MINGW" ] || [ "$(expr substr $(uname -s) 1 10)" == "MSYS_NT-10" ]; then
    source venv/Scripts/activate
fi

# Install requirements
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Run the app
echo "Setup complete! Run the app with: streamlit run app.py" 