#!/usr/bin/env python3
"""
Helper script to install dlib, which can be challenging on different platforms.
This script tries multiple approaches based on the operating system.
"""
import os
import sys
import platform
import subprocess

def install_dlib():
    """Install dlib package using the best method for the current platform."""
    system = platform.system()
    print(f"Detected operating system: {system}")
    
    # Try installing with pip first (default method)
    print("Attempting to install dlib with pip...")
    
    try:
        # Check if dlib is already installed
        import dlib
        print("dlib is already installed!")
        return True
    except ImportError:
        pass
    
    pip_install_cmd = [sys.executable, "-m", "pip", "install", "dlib==19.24.2"]
    
    try:
        subprocess.check_call(pip_install_cmd)
        print("Successfully installed dlib with pip!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install dlib with pip. Trying alternative methods...")
    
    if system == "Windows":
        # On Windows, try to install from a wheel
        print("Trying to install dlib from wheel...")
        
        python_version = f"{sys.version_info.major}{sys.version_info.minor}"
        wheel_name = None
        
        # Try to find a wheel that matches the Python version
        wheel_files = [f for f in os.listdir(".") if f.endswith(".whl") and "dlib" in f]
        for wheel in wheel_files:
            if f"cp{python_version}" in wheel:
                wheel_name = wheel
                break
        
        if wheel_name:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_name])
                print(f"Successfully installed dlib from wheel: {wheel_name}")
                return True
            except subprocess.CalledProcessError:
                print(f"Failed to install dlib from wheel: {wheel_name}")
        else:
            print(f"No matching dlib wheel found for Python {sys.version_info.major}.{sys.version_info.minor}")
            
            # Download a wheel from a trusted source
            print("Trying to download a pre-built wheel...")
            wheel_url = f"https://github.com/z-mahmud22/Dlib_Windows_Python3x/raw/main/dlib-19.24.0-cp{python_version}-cp{python_version}-win_amd64.whl"
            try:
                subprocess.check_call(["curl", "-L", wheel_url, "-o", "dlib_wheel.whl"])
                subprocess.check_call([sys.executable, "-m", "pip", "install", "dlib_wheel.whl"])
                print("Successfully installed dlib from downloaded wheel!")
                return True
            except subprocess.CalledProcessError:
                print("Failed to download or install dlib wheel.")
    
    elif system == "Linux":
        # On Linux, try installing system dependencies first
        print("Installing system dependencies for dlib...")
        try:
            # Check if we're on a Debian-based system
            subprocess.check_call(["apt-get", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            is_debian = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            is_debian = False
            
        if is_debian:
            try:
                subprocess.check_call(["sudo", "apt-get", "update"])
                subprocess.check_call([
                    "sudo", "apt-get", "install", "-y",
                    "cmake", "build-essential", "libopenblas-dev", "liblapack-dev", "libx11-dev",
                    "libgtk-3-dev", "python3-dev"
                ])
            except subprocess.CalledProcessError:
                print("Failed to install system dependencies. You may need to install them manually.")
                
        # Try installing dlib with pip again
        try:
            subprocess.check_call(pip_install_cmd)
            print("Successfully installed dlib after installing system dependencies!")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install dlib with pip even after installing dependencies.")
    
    elif system == "Darwin":  # macOS
        # For macOS, try with Homebrew
        print("Installing dependencies via Homebrew...")
        try:
            # Check if Homebrew is installed
            subprocess.check_call(["brew", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Install dependencies
            subprocess.check_call(["brew", "install", "cmake"])
            subprocess.check_call(["brew", "install", "boost"])
            subprocess.check_call(["brew", "install", "boost-python3"])
            
            # Try installing dlib with pip again
            subprocess.check_call(pip_install_cmd)
            print("Successfully installed dlib after installing Homebrew dependencies!")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Homebrew not found or failed to install dependencies.")
    
    # If all automated methods failed, provide instructions for manual installation
    print("\n==== MANUAL INSTALLATION REQUIRED ====")
    print("Automated dlib installation failed. Please try manual installation:")
    print("\n1. Ensure you have CMake installed:")
    print("   - Windows: Download from https://cmake.org/download/")
    print("   - Linux: sudo apt-get install cmake build-essential")
    print("   - macOS: brew install cmake")
    print("\n2. Clone and build dlib:")
    print("   git clone https://github.com/davisking/dlib.git")
    print("   cd dlib")
    print("   python setup.py install")
    print("\nFor more details, visit: https://github.com/davisking/dlib")
    
    return False

if __name__ == "__main__":
    success = install_dlib()
    sys.exit(0 if success else 1) 