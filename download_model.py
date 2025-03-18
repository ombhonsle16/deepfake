import os
import urllib.request

def download_file(url, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    print(f"Downloading {url} to {destination}...")
    urllib.request.urlretrieve(url, destination)
    print(f"Downloaded to {destination}")

if __name__ == "__main__":
    model_url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
    destination_path = "models/shape_predictor_68_face_landmarks.dat"
    download_file(model_url, destination_path) 