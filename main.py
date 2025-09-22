# backend/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from utils.extract_frames import extract_frames
import timm  # ✅ SA-Xception 백본용

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# SA-Xception 정의 (학습 코드와 동일)
# ---------------------------
class SA_Xception(nn.Module):
    def __init__(self, base_model='xception', num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(base_model, pretrained=False, num_classes=0)
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        feat = self.backbone.forward_features(x)  # (B, 2048, H, W)
        attn = self.attention(feat)               # (B, 1, H, W)
        weighted = feat * attn
        pooled = self.pool(weighted).view(x.size(0), -1)
        return self.fc(pooled)                    # (B, 2)

# ---------------------------
# 모델 로딩
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "checkpoints/x_best_sa_xception.pth"  # ✅ 로컬 체크포인트 사용
assert os.path.exists(MODEL_PATH), f"모델 파일이 없습니다: {MODEL_PATH}"

model = SA_Xception(num_classes=2)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

# state_dict/전체저장 모두 대응 + 'module.' 접두사 제거
if isinstance(ckpt, dict) and ("state_dict" in ckpt or any(isinstance(k, str) and k.startswith("module.") for k in ckpt.keys())):
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load_state_dict] missing={missing}, unexpected={unexpected}")
elif isinstance(ckpt, dict):
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print(f"[load_state_dict] missing={missing}, unexpected={unexpected}")
else:
    model = ckpt  # 전체 모델 저장된 형태

model.eval().to(DEVICE)

# ---------------------------
# 전처리 정의 (학습과 동일)
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

CLASS_NAMES = ["REAL", "FAKE"]  # ✅ 학습 데이터셋 라벨 순서에 맞춰 필요시 ["FAKE","REAL"]로 바꾸세요.
FAKE_IDX = 1                    # ✅ 위 CLASS_NAMES 기준 FAKE 인덱스

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

    os.makedirs(temp_frame_dir, exist_ok=True)
    num_frames = extract_frames(temp_video_path, temp_frame_dir, max_frames=32)
    print(f"📸 추출된 프레임 수: {num_frames}")

    if num_frames == 0:
        # 정리
        try:
            os.remove(temp_video_path)
            shutil.rmtree(temp_frame_dir)
        except Exception:
            pass
        return JSONResponse(status_code=400, content={"error": "프레임 추출 실패"})

    try:
        # 기존 로직 유지: 프레임 텐서 평균 → 1장처럼 처리
        inputs = []
        for i in range(num_frames):
            frame_path = os.path.join(temp_frame_dir, f"frame_{i:03d}.jpg")
            img = Image.open(frame_path).convert("RGB")
            inputs.append(transform(img))
        input_tensor = sum(inputs) / len(inputs)
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)  # (1, 3, 299, 299)

        with torch.no_grad():
            logits = model(input_tensor)                     # (1, 2)
            probs = torch.softmax(logits, dim=1).squeeze(0) # (2,)
            prob_fake = probs[FAKE_IDX].item()
            pred_idx = int(torch.argmax(probs).item())

        print(f"🧠 예측 결과: probs={probs.tolist()} -> pred={CLASS_NAMES[pred_idx]}")
    except Exception as e:
        print(f"❌ 예측 중 오류 발생: {e}")
        # 정리
        try:
            os.remove(temp_video_path)
            shutil.rmtree(temp_frame_dir)
        except Exception:
            pass
        return JSONResponse(status_code=500, content={"error": "추론 실패", "detail": str(e)})

    # 정리
    try:
        os.remove(temp_video_path)
        shutil.rmtree(temp_frame_dir)
    except Exception:
        pass

    return {
        "class_probabilities": {CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))},
        "deepfake_probability": round(prob_fake * 100, 2),
        "prediction": CLASS_NAMES[pred_idx]
    }

@app.get("/")
def health_check():
    return {"status": "✅ Deepfake API is running.", "device": str(DEVICE), "model": "SA-Xception", "checkpoint": MODEL_PATH}
