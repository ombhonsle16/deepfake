#!/bin/bash

echo "==================================="
echo "Advanced Deepfake Detection System"
echo "==================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or later."
    echo "Visit https://www.python.org/downloads/"
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

# Make this script executable
chmod +x "$0"

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

# Activate venv and install requirements
echo "Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if ! python -c "import streamlit" &> /dev/null; then
    echo "Installing requirements..."
    
    # First try to install dlib using our helper script
    echo "Installing dlib (this might take a moment)..."
    python install_dlib.py
    
    # Then install other dependencies
    echo "Installing other dependencies..."
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "Failed to install requirements."
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

# Download model files if they don't exist
if [ ! -f "models/shape_predictor_68_face_landmarks.dat" ]; then
    echo "Downloading model files..."
    python download_model.py
    if [ $? -ne 0 ]; then
        echo "Failed to download model files."
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

# Make models directory if it doesn't exist
if [ ! -d "models" ]; then
    mkdir -p models
fi

# Run the app
echo
echo "Starting Deepfake Detection System..."
echo
echo "The application will open in your default web browser."
echo
echo "IMPORTANT: Do not close this window while using the application."
echo
cd src && python -m streamlit run mock_app.py

read -p "Press Enter to exit..." 