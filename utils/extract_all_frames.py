import os
from extract_frames import extract_frames

video_root = "../../sample_videos/train_sample_videos"
output_root = "../../extracted_frames"
max_frames = 32

os.makedirs(output_root, exist_ok=True)

video_files = [f for f in os.listdir(video_root) if f.endswith(".mp4")]
print(f"🔍 총 {len(video_files)}개의 영상 발견")

for i, video in enumerate(video_files):
    name = os.path.splitext(video)[0]
    video_path = os.path.join(video_root, video)
    output_dir = os.path.join(output_root, name)

    if os.path.exists(os.path.join(output_dir, "frame_000.jpg")):
        print(f"✅ {name}: 이미 추출됨, 건너뜀")
        continue

    try:
        count = extract_frames(video_path, output_dir, max_frames=max_frames)
        print(f"[{i+1}/{len(video_files)}] ✅ {name}: {count}프레임 추출")
    except Exception as e:
        print(f"[{i+1}/{len(video_files)}] ❌ {name}: 오류 - {e}")
