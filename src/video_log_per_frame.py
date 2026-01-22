import cv2
from ultralytics import YOLO
import sys
import os
from datetime import datetime, timedelta

# 1. 모델 및 영상 경로 설정
model_path = os.getenv("YOLO_MODEL_PATH")
video_path = os.getenv("VIDEO_PATH")
output_dir = os.getenv("OUTPUT_DIR")
output_log_path = os.getenv("OUTPUT_LOG_PATH")

if not model_path or not video_path or not output_dir or not output_log_path:
    print(
        "환경 변수 YOLO_MODEL_PATH, VIDEO_PATH, OUTPUT_DIR, OUTPUT_LOG_PATH를 설정해주세요."
    )
    sys.exit()

print("모델 경로:", model_path)
print("영상 경로:", video_path)
print("출력 디렉토리:", output_dir)
print("출력 로그 파일 경로:", output_log_path)
os.makedirs(output_dir, exist_ok=True)
model = YOLO(model_path)

# 2. 영상 정보 읽기
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("영상을 열 수 없습니다.")
    sys.exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 3. 기준 시간 설정 (오전 8시 0분 0초)
start_real_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)

# 중복 저장 방지용 변수 (초 단위)
last_logged_minute = None

# 로그 파일 열기
with open(output_log_path, "w", encoding="utf-8") as f:
    # 헤더 작성
    f.write("=== 벌 탐지 로그 (분 단위) ===\n")
    f.write(f"분석 파일: {video_path}\n")
    f.write(f"기준 시작 시간: {start_real_time.strftime('%H:%M:%S')}\n")
    f.write("-" * 70 + "\n")
    f.write(
        f" {'실제 시간(분)':^10} | {'영상 위치':^10} | {'개수':^5} | {'저장된 이미지 파일명':^25}\n"
    )
    f.write("-" * 70 + "\n")

    # 추론 시작
    results = model.predict(source=video_path, stream=True, verbose=False)

    for i, result in enumerate(results):

        boxes = result.boxes

        # --------------------------------------------------
        # 1. 로그 기록
        # --------------------------------------------------

        # 박스 정보 나열
        log_line = ""
        if boxes:
            for box in boxes:
                x, y, w, h = map(int, box.xywh[0])
                log_line += f"{x} {y} {w} {h} "
        log_line += "\n"

        # 로그 파일에 쓰기
        f.write(log_line)
        f.flush()

        # 진행률 표시 (1000프레임마다)
        if i % 1000 == 0:
            progress = (i / total_frames) * 100
            print(f"... {progress:.2f}% 진행 중", end="\r")

cap.release()
print(f"생성된 로그 파일: {output_log_path}")
