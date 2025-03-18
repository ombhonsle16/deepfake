# Installation Guide for Deepfake Detection System

This guide will help you set up the deepfake detection system on your Windows machine.

## Prerequisites

- Python 3.8+ (Python 3.13 is supported)
- Git (optional, for cloning the repository)
- Windows 10 or later
- Microsoft Visual C++ Redistributable (will be installed automatically if needed)

## Step 1: Clone or Download the Repository

If you have Git installed:

```bash
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection
```

Alternatively, download and extract the ZIP file from the repository.

## Step 2: Install Microsoft Visual C++ Build Tools (Required)

Some Python packages require Microsoft Visual C++ Build Tools to install. You can download and install them from:

[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

During installation, make sure to select "Desktop development with C++" workload.

## Step 3: Create a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

## Step 4: Install Dependencies

### Easiest Method: Use the Installation Script

Run the provided installation script:

```bash
install_windows.bat
```

This script will automatically set up the environment and install all dependencies.

### Manual Installation

If you prefer to install manually:

1. Update pip to the latest version:

```bash
python -m pip install --upgrade pip
```

2. Install PyTorch:

```bash
# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If you have a compatible NVIDIA GPU and want to use CUDA:

```bash
# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. Install NumPy (for Python 3.13, use version 2.2.4 or newer):

```bash
pip install "numpy==2.2.4" --only-binary=numpy
```

4. Install scikit-learn (for Python 3.13, use version 1.6.1 or newer):

```bash
pip install "scikit-learn==1.6.1" --only-binary=scikit-learn
```

5. Install OpenCV (required for image processing):

```bash
pip install opencv-python
```

6. Install matplotlib (required for visualization):

```bash
pip install matplotlib
```

7. Install MTCNN (required for face detection):

```bash
pip install mtcnn
```

8. Install ONNX and ONNX Runtime:

For Python 3.13:
```bash
# ONNX Runtime 1.20.0+ supports Python 3.13
pip install "onnxruntime>=1.20.0" --only-binary=onnxruntime

# ONNX 1.15.0 is not compatible with Python 3.13
# You can try the weekly development build
pip install onnx-weekly --only-binary=onnx-weekly
```

For Python 3.8-3.12:
```bash
pip install "onnx==1.15.0" --only-binary=onnx
pip install "onnxruntime>=1.17.0" --only-binary=onnxruntime
```

9. Install Streamlit:

```bash
pip install streamlit
```

10. Install the remaining dependencies:

```bash
pip install -r requirements.txt --only-binary=:all: --prefer-binary
```

## Step 5: Optional Dependencies

Some features require additional setup:

### Face Detection with MTCNN

MTCNN should be installed automatically from requirements.txt, but if you encounter issues:

```bash
pip install mtcnn==0.1.1
```

### dlib (Optional)

dlib is optional and can be difficult to install on Windows. If you want to use it:

1. Install Visual Studio Build Tools with C++ support (see Step 2)
2. Install CMake
3. Run:
   ```bash
   pip install dlib
   ```

You can skip dlib installation as the system uses MTCNN by default.

## Step 6: Test the Installation

Run the Streamlit web interface:

```bash
# Make sure your virtual environment is activated
venv\Scripts\activate

# Run the application using the Python module
python -m streamlit run src/app.py
```

This should open a web browser with the deepfake detection interface.

## Troubleshooting

### "Microsoft Visual C++ 14.0 or greater is required"

This error occurs when trying to build packages from source. You need to:

1. Install Microsoft Visual C++ Build Tools from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. During installation, select "Desktop development with C++" workload
3. Try installing again after the installation completes

Alternatively, use the `--only-binary=:all:` flag to avoid building from source:

```bash
pip install -r requirements.txt --only-binary=:all: --prefer-binary
```

### "Could not find a version that satisfies the requirement"

This error occurs when the specified package version isn't available for your Python version. Try:

1. Using a newer version of the package (e.g., for scikit-learn on Python 3.13, use 1.6.1 instead of 1.4.1)
2. Installing without specifying a version to get the latest compatible version:
   ```bash
   pip install scikit-learn --only-binary=scikit-learn
   ```

### "No module named 'cv2'"

If you encounter this error, OpenCV is not installed. Install it with:

```bash
pip install opencv-python
```

### "No module named 'matplotlib'"

If you encounter this error, matplotlib is not installed. Install it with:

```bash
pip install matplotlib
```

### "No module named 'mtcnn'"

If you encounter this error, MTCNN is not installed. Install it with:

```bash
pip install mtcnn
```

### ONNX Compatibility with Python 3.13

If you're using Python 3.13 and encounter issues with ONNX:

1. ONNX 1.15.0 is not compatible with Python 3.13. Full support will come in ONNX 1.18 (not yet released).
2. You can use ONNX Runtime 1.20.0 or newer which does support Python 3.13.
3. For ONNX functionality, you can try the development build:
   ```bash
   pip install onnx-weekly --only-binary=onnx-weekly
   ```
4. The system will still function with limited ONNX capabilities using only ONNX Runtime.

### Streamlit Command Not Found

If you encounter an error where `streamlit` is not recognized as a command:

1. Make sure your virtual environment is activated: `venv\Scripts\activate`
2. Run Streamlit as a Python module instead: `python -m streamlit run src/app.py`
3. If that doesn't work, try reinstalling Streamlit: `pip install streamlit`

### ImportError: DLL load failed

This usually happens with OpenCV or other libraries that depend on Visual C++ redistributable packages. Install the latest [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

### Build errors with NumPy or other packages

If you encounter build errors, it's usually because the package is trying to build from source. Try installing pre-built wheels:

```bash
pip install package_name --only-binary=package_name
```

For NumPy specifically on Python 3.13, use version 2.2.4 or newer:

```bash
pip install "numpy==2.2.4" --only-binary=numpy
```

### CUDA-related errors

If you're using a GPU and encounter CUDA errors:

1. Make sure your GPU drivers are up to date
2. Verify that the CUDA version matches your PyTorch installation
3. Try the CPU-only version if problems persist

### Other Issues

If you encounter other issues, please check the project's GitHub issues page or create a new issue with details about your problem. 