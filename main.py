# backend/main.py
from utils.download import maybe_download_model
maybe_download_model()

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
import torch
from torchvision.models import efficientnet_b0
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from utils.extract_frames import extract_frames

from fastapi.middleware.cors import CORSMiddleware

import os
import requests

# 구글 드라이브 다운로드 함수
def download_from_google_drive(url, output_path):
    response = requests.get(url, allow_redirects=True)
    with open(output_path, 'wb') as f:
        f.write(response.content)

# .pth 파일이 없으면 다운로드
pth_path = "checkpoints/deepfake_efficientnet.pth"
if not os.path.exists(pth_path):
    os.makedirs("checkpoints", exist_ok=True)
    print("🧩 Downloading model...")
    download_from_google_drive(
        "https://drive.google.com/uc?export=download&id=1596t3TegPKwnaKGTk2z7YnbJGeGUuO6A",  # 공유 링크에서 id만 추출
        pth_path
    )



app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 모델 로딩
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model.load_state_dict(torch.load("checkpoints/deepfake_efficientnet.pth", map_location=DEVICE))
model.eval().to(DEVICE)

# ---------------------------
# 전처리 정의
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------
# API 엔드포인트
# ---------------------------
@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    temp_video_path = "temp_video.mp4"
    temp_frame_dir = "temp_frames"

    with open(temp_video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print("✅ 업로드 파일 저장 완료")

    # 프레임 추출
    os.makedirs(temp_frame_dir, exist_ok=True)
    num_frames = extract_frames(temp_video_path, temp_frame_dir, max_frames=32)
    print(f"📸 추출된 프레임 수: {num_frames}")

    if num_frames == 0:
        print("❌ 프레임 추출 실패")
        return JSONResponse(status_code=400, content={"error": "프레임 추출 실패"})

    try:
        inputs = []
        for i in range(num_frames):
            frame_path = os.path.join(temp_frame_dir, f"frame_{i:03d}.jpg")
            img = Image.open(frame_path).convert("RGB")
            inputs.append(transform(img))
        input_tensor = sum(inputs) / len(inputs)
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()

        print(f"🧠 예측 결과: {prob:.4f}")
    except Exception as e:
        print(f"❌ 예측 중 오류 발생: {e}")
        return JSONResponse(status_code=500, content={"error": "추론 실패"})

    os.remove(temp_video_path)
    shutil.rmtree(temp_frame_dir)

    return {
        "deepfake_probability": round(prob * 100, 2),
        "prediction": "FAKE" if prob > 0.5 else "REAL"
    }
