# backend/utils/extract_frames.py

import cv2
import os

def extract_frames(video_path, output_dir, max_frames=32):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)

    count, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or saved >= max_frames:
            break
        if count % step == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1

    cap.release()
    print(f"✅ Extracted {saved} frames to {output_dir}")
    return saved
