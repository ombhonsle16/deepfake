import os
import urllib.request
import sys

def download_file(url, destination):
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Check if file already exists
        if os.path.exists(destination):
            print(f"File already exists at {destination}, skipping download")
            return True
            
        print(f"Downloading {url} to {destination}...")
        urllib.request.urlretrieve(url, destination)
        print(f"Downloaded to {destination}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # 1. Face landmark predictor model (already included in install scripts, but added as fallback)
    face_model_url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
    face_model_path = "models/shape_predictor_68_face_landmarks.dat"
    
    # Only download if it doesn't exist (since install scripts already handle this)
    print("Checking face landmark predictor model...")
    if not os.path.exists(face_model_path):
        print("Downloading face landmark predictor model...")
        download_file(face_model_url, face_model_path)
    else:
        print(f"Face landmark model already exists at {face_model_path}")

    # 2. Pre-trained EfficientNet weights
    efficientnet_url = "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth"
    efficientnet_path = "models/efficientnet-b0.pth"
    
    print("Checking pre-trained EfficientNet weights...")
    download_file(efficientnet_url, efficientnet_path)
    
    # 3. Vision Transformer weights
    vit_url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth"
    vit_path = "models/vit_base_p16_224.pth"
    
    print("Checking Vision Transformer weights...")
    download_file(vit_url, vit_path)
    
    print("\nAll model downloads completed!")

if __name__ == "__main__":
    main() 