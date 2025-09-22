import os, shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import onnxruntime as ort
from utils.extract_frames import extract_frames
import torch  # 전처리/softmax 편의용

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# 모델 경로 (양자화했으면 int8 onnx를 지정)
BASE = os.path.dirname(os.path.abspath(__file__))
CAND = [
    os.getenv("MODEL_PATH","").strip(),
    os.path.join(BASE, "checkpoints", "sa_xception.int8.onnx"),
    os.path.join(BASE, "checkpoints", "sa_xception.onnx"),
    "/app/checkpoints/sa_xception.int8.onnx",
    "/app/checkpoints/sa_xception.onnx",
]
MODEL_PATH = next((p for p in CAND if p and os.path.exists(p)), None)
assert MODEL_PATH, "ONNX 모델을 찾을 수 없습니다."

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

CLASS_NAMES = ["REAL","FAKE"]
FAKE_IDX = 1

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    temp_video_path = "temp_video.mp4"
    temp_frame_dir = "temp_frames"
    with open(temp_video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    os.makedirs(temp_frame_dir, exist_ok=True)
    n = extract_frames(temp_video_path, temp_frame_dir, max_frames=32)
    if n == 0:
        cleanup(temp_video_path, temp_frame_dir)
        return JSONResponse(status_code=400, content={"error":"프레임 추출 실패"})

    try:
        frames = []
        for i in range(n):
            img = Image.open(os.path.join(temp_frame_dir, f"frame_{i:03d}.jpg")).convert("RGB")
            frames.append(transform(img))
        # 기존과 동일: 프레임 평균 → 1장처럼
        x = (sum(frames)/len(frames)).unsqueeze(0).numpy().astype(np.float32)  # (1,3,299,299)
        logits = sess.run(None, {"input": x})[0]  # (1,2)
        probs = torch.softmax(torch.from_numpy(logits), dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        prob_fake = float(probs[FAKE_IDX].item())
        out = {
            "class_probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
            "deepfake_probability": round(prob_fake*100, 2),
            "prediction": CLASS_NAMES[pred_idx]
        }
        return out
    except Exception as e:
        return JSONResponse(status_code=500, content={"error":"추론 실패", "detail": str(e)})
    finally:
        cleanup(temp_video_path, temp_frame_dir)

def cleanup(video, frame_dir):
    try:
        if os.path.exists(video): os.remove(video)
        if os.path.exists(frame_dir): shutil.rmtree(frame_dir)
    except: pass

@app.get("/")
def health(): return {"status":"ok", "model": os.path.basename(MODEL_PATH)}
