FROM python:3.10-slim

# ffmpeg만 있으면 됨 (opencv-python-headless 사용 가정)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code
COPY main.py .
COPY utils/ utils/

# ✅ 모델 가중치 전체 복사 (혹은 특정 파일만 명시적으로 복사)
# COPY checkpoints/x_best_sa_xception.pth /app/checkpoints/x_best_sa_xception.pth
COPY checkpoints/ /app/checkpoints/

# (선택) 빌드 타임 확인
RUN ls -al /app/checkpoints || true

# ✅ Render는 동적으로 PORT를 줍니다
ENV PORT=10000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
