import cv2
from ultralytics import YOLO
import sys
import os
from datetime import datetime, timedelta

# 1. 모델 및 영상 경로 설정
model = YOLO("/home/berry/WCL_bee/yolo_bee_detect/runs/detect/train21/weights/best.pt")
video_path = "/mnt/d/bee/cut_output_merged_6_0108.mp4"
# video_path = "/mnt/c/Users/berry/Downloads/KakaoTalk_20251219_165218533.mp4"
# output_video_path = "/mnt/c/Users/berry/Desktop/mybee/output_video_csv/output_with_count_2.mp4"
output_dir = "/mnt/d/bee/cut_output_merged_6_0108"
os.makedirs(output_dir, exist_ok=True)
output_log_path = "/mnt/d/bee/cut_output_merged_6_0108/cut_output_merged_6_0108_log.txt"

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
with open(output_log_path, 'w', encoding='utf-8') as f:
    # 헤더 작성
    f.write("=== 벌 탐지 로그 (분 단위) ===\n")
    f.write(f"분석 파일: {video_path}\n")
    f.write(f"기준 시작 시간: {start_real_time.strftime('%H:%M:%S')}\n")
    f.write("-" * 70 + "\n")
    f.write(f" {'실제 시간(분)':^10} | {'영상 위치':^10} | {'개수':^5} | {'저장된 이미지 파일명':^25}\n")
    f.write("-" * 70 + "\n")

    # 추론 시작
    results = model.predict(source=video_path, stream=True, verbose=False)

    for i, result in enumerate(results):
        
        boxes = result.boxes
        # 벌이 1마리 이상 감지되었을 때만 로직 수행
        if len(boxes) > 0:
            
            # 시간 계산
            elapsed_seconds = i / fps
            current_real_time = start_real_time + timedelta(seconds=elapsed_seconds)
            
            # "시:분" 형식 문자열 (예: "07:05") -> 분 단위 체크용
            current_minute_str = current_real_time.strftime("%H:%M")
            
            # ★ 핵심 로직: 이번 '분(Minute)'이 이전에 기록한 '분'과 다를 때만 실행 ★
            # 즉, 7시 5분에 벌이 계속 보여도 딱 한 번만 실행됨
            if current_minute_str != last_logged_minute:
                
                # --------------------------------------------------
                # 1. 이미지 저장 (분당 1회)
                # --------------------------------------------------
                # 파일명용 시간 (초 단위까지 포함해서 파일명 생성)
                time_filename = current_real_time.strftime("%H_%M_%S")
                save_filename = f"{time_filename}_capture.jpg"
                save_path = os.path.join(output_dir, save_filename)
                
                # 박스 그려진 이미지 생성 및 시간 자막 추가
                annotated_frame = result.plot()
                time_display = current_real_time.strftime("%H:%M:%S")
                cv2.putText(annotated_frame, f"Time: {time_display}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                # 파일 저장
                cv2.imwrite(save_path, annotated_frame)

                # --------------------------------------------------
                # 2. 로그 기록
                # --------------------------------------------------
                # 영상 위치 계산
                v_total_sec = int(elapsed_seconds)
                v_h, v_m, v_s = v_total_sec // 3600, (v_total_sec % 3600) // 60, v_total_sec % 60
                video_time_str = f"{v_h:02}:{v_m:02}:{v_s:02}"
                
                # 로그 파일에 쓰기
                log_line = f" [{current_minute_str}]   |  {video_time_str}  |  {len(boxes):^3}  |  {save_filename}\n"
                f.write(log_line)
                f.flush() 
                
                # 콘솔 출력
                print(f"📸 캡처 & 기록: {log_line.strip()}")
                
                # 중복 방지 변수 업데이트
                last_logged_minute = current_minute_str
        
        # 진행률 표시 (1000프레임마다)
        if i % 1000 == 0:
            progress = (i / total_frames) * 100
            print(f"... {progress:.1f}% 진행 중")

cap.release()
print(f"생성된 로그 파일: {output_log_path}")