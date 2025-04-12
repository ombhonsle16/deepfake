@echo off
echo ===================================
echo Advanced Deepfake Detection System
echo ===================================
echo.

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed. Please install Python 3.8 or later.
    echo Visit https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: Check if venv exists, if not create it
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate venv and install requirements
echo Activating virtual environment...
call venv\Scripts\activate

:: Check if requirements are installed
if not exist venv\Lib\site-packages\streamlit (
    echo Installing requirements...
    
    :: First try to install dlib using our helper script
    echo Installing dlib (this might take a moment)...
    python install_dlib.py
    
    :: Then install other dependencies
    echo Installing other dependencies...
    pip install -r requirements.txt
    
    if %ERRORLEVEL% neq 0 (
        echo Failed to install requirements.
        pause
        exit /b 1
    )
)

:: Download model files if they don't exist
if not exist models\shape_predictor_68_face_landmarks.dat (
    echo Downloading model files...
    python download_model.py
    if %ERRORLEVEL% neq 0 (
        echo Failed to download model files.
        pause
        exit /b 1
    )
)

:: Run the app
echo.
echo Starting Deepfake Detection System...
echo.
echo The application will open in your default web browser.
echo.
echo IMPORTANT: Do not close this window while using the application.
echo.
cd src && python -m streamlit run mock_app.py

pause 