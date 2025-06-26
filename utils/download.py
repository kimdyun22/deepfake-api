import os
import requests

def download_from_google_drive(url, output_path):
    print(f"📥 Downloading model from {url}...")
    response = requests.get(url, allow_redirects=True)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print("✅ Download complete.")
    else:
        raise RuntimeError(f"❌ Download failed. Status code: {response.status_code}")

def maybe_download_model():
    model_path = "checkpoints/deepfake_efficientnet.pth"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?export=download&id=1596t3TegPKwnaKGTk2z7YnbJGeGUuO6A"
        download_from_google_drive(url, model_path)

