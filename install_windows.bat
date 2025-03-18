@echo off
setlocal enabledelayedexpansion

echo Starting Deepfake Detection System Installation...

:: Check Python installation
python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.13 or later.
    exit /b 1
)

:: Create and activate virtual environment
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install core dependencies
echo Installing core dependencies...
pip install numpy==1.26.3

:: Install PyTorch (latest version compatible with Python 3.13)
echo Installing PyTorch (latest compatible version)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

:: Install scikit-learn
echo Installing scikit-learn...
pip install scikit-learn==1.6.1 --only-binary=scikit-learn

:: Install OpenCV
echo Installing OpenCV...
pip install opencv-python==4.9.0.80

:: Install pre-built packages for SciPy
echo Installing SciPy...
pip install scipy --only-binary=scipy

:: Install dlib
echo Installing dlib...
pip install dlib==19.24.2

:: Create models directory
if not exist "models" mkdir models

:: Download face landmark predictor model
echo Downloading face landmark predictor model...
powershell -Command "& {Invoke-WebRequest -Uri 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2' -OutFile 'models/shape_predictor_68_face_landmarks.dat.bz2'}"

:: Extract using Python since bunzip2 is not available
echo Extracting face landmark model...
python -c "import bz2; open('models/shape_predictor_68_face_landmarks.dat', 'wb').write(bz2.BZ2File('models/shape_predictor_68_face_landmarks.dat.bz2').read())"

:: Install simplified transformers (without tokenizers)
echo Installing transformers (simplified)...
pip install transformers --no-deps
pip install filelock huggingface-hub packaging pyyaml requests

:: Install Streamlit and other UI dependencies
echo Installing UI dependencies...
pip install streamlit==1.31.1 plotly==5.18.0

:: Install image processing libraries
echo Installing image processing libraries...
pip install Pillow==10.2.0 matplotlib==3.8.3

:: Install utilities
echo Installing utilities...
pip install tqdm==4.66.2 pandas==2.2.1

echo Installation completed successfully!
echo.
echo To run the application:
echo 1. Activate the virtual environment: venv\Scripts\activate
echo 2. Run the application: python -m streamlit run src/mock_app.py
echo.
pause 