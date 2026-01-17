import cv2
from ultralytics import YOLO
import sys
import os
import math
from datetime import datetime, timedelta

# ==============================================================================
# 1. 설정 및 경로
# ==============================================================================
model_path = "/home/berry/WCL_bee/yolo_bee_detect/models/train_20251231/weights/best.pt"
video_path = "/mnt/e/tmp/mp4_files/cut_output_merged_15.mp4"

# 결과 저장 경로
output_dir = "/mnt/c/Users/berry/Desktop/mybee/output_video_csv/cut_output_merged_15_entrance"
os.makedirs(output_dir, exist_ok=True)
output_log_path = "/mnt/c/Users/berry/Desktop/mybee/output_video_csv/cut_output_merged_15_entrance/cut_output_merged_15_log_entrance.txt"

# ------------------------------------------------------------------------------
# ★ 카운팅 기준선 설정 (영상 해상도에 맞춰 조절 필요) ★
# ------------------------------------------------------------------------------
# Y좌표 (높이), in/out
LINE_IN_Y = 1050  # 파란선 (위 -> 아래 통과 시 IN)
LINE_OUT_Y = 1075  # 빨간선 (아래 -> 위 통과 시 OUT)
# 1150/1200: 2, 17
# 1125/1175: 4
# 1100/1150: 18
# 1100/1125: 14
# 1050/1100: 3
# 1050/1075: 15

# X좌표 (가로 폭 제한)
LINE_X_START = 40
LINE_X_END = 1600

# 오차 범위 및 매칭 거리
OFFSET = 0
MAX_DIST_MATCH = 100
# ==============================================================================

# 2. 모델 및 영상 로드
print("모델 및 영상 로딩 중...")
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상을 열 수 없습니다.")
    sys.exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 3. 기준 시간 설정 (오전 8시 0분 0초)
start_real_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)

# 변수 초기화
last_logged_minute = None
count_in = 0
count_out = 0
prev_bees = [] # 이전 프레임 벌 정보

print(f"분석 시작... (총 {total_frames} 프레임)")
print(f"로그 파일: {output_log_path}")

# 로그 파일 열기
with open(output_log_path, 'w', encoding='utf-8') as f:
    f.write("=== 벌 출입(Event) 감지 로그 (분 단위) ===\n")
    f.write(f"분석 파일: {video_path}\n")
    f.write(f"기준 시작 시간: {start_real_time.strftime('%H:%M:%S')}\n")
    f.write("-" * 100 + "\n")
    f.write(f" {'실제 시간':^10} | {'이벤트':^5} | {'IN누적':^5} | {'OUT누적':^5} | {'저장된 이미지 파일명':^30}\n")
    f.write("-" * 100 + "\n")

    # 추론 시작
    results = model.predict(source=video_path, stream=True, verbose=False)

    for i, result in enumerate(results):
        
        boxes = result.boxes
        current_bees = []
        
        # 이번 프레임에서 출입이 발생했는지 체크하는 플래그
        event_occurred = False
        event_type = "" # "IN" 또는 "OUT"

        # ----------------------------------------------------------
        # [Step A] 데이터 추출 및 출입 카운팅
        # ----------------------------------------------------------
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            current_bees.append({
                'center': (cx, cy),
                'bottom': int(y2)
            })

        if prev_bees:
            matched_indices = set()
            
            for curr_bee in current_bees:
                curr_cx, curr_cy = curr_bee['center']
                
                # X범위 필터링
                if not (LINE_X_START <= curr_cx <= LINE_X_END):
                    continue

                # 거리 기반 매칭
                min_dist = MAX_DIST_MATCH
                best_prev_bee = None
                best_prev_idx = -1
                
                for idx, prev_bee in enumerate(prev_bees):
                    if idx in matched_indices: continue
                    prev_cx, prev_cy = prev_bee['center']
                    dist = math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_prev_bee = prev_bee
                        best_prev_idx = idx
                
                if best_prev_bee is not None:
                    matched_indices.add(best_prev_idx)
                    prev_bottom = best_prev_bee['bottom']
                    curr_bottom = curr_bee['bottom']
                    
                    # IN: 위 -> 아래
                    if prev_bottom < (LINE_IN_Y - OFFSET) and curr_bottom > (LINE_IN_Y + OFFSET):
                        count_in += 1
                        event_occurred = True
                        event_type = "IN"
                    
                    # OUT: 아래 -> 위
                    elif prev_bottom > (LINE_OUT_Y + OFFSET) and curr_bottom < (LINE_OUT_Y - OFFSET):
                        count_out += 1
                        event_occurred = True
                        event_type = "OUT"

        # ----------------------------------------------------------
        # [Step B] 출입 이벤트 발생 시에만 로그 및 저장
        # ----------------------------------------------------------
        if event_occurred:
            
            elapsed_seconds = i / fps
            current_real_time = start_real_time + timedelta(seconds=elapsed_seconds)
            current_minute_str = current_real_time.strftime("%H:%M")
            
            # ★ 핵심: 이번 분(Minute)에 아직 기록하지 않았다면 저장 ★
            if current_minute_str != last_logged_minute:
                
                # 1. 이미지 생성
                annotated_frame = result.plot()
                
                # 기준선 그리기
                cv2.line(annotated_frame, (LINE_X_START, LINE_IN_Y), (LINE_X_END, LINE_IN_Y), (255, 0, 0), 2)
                cv2.line(annotated_frame, (LINE_X_START, LINE_OUT_Y), (LINE_X_END, LINE_OUT_Y), (0, 0, 255), 2)
                
                # 자막 추가
                time_display = current_real_time.strftime("%H:%M:%S")
                # 어떤 이벤트였는지 표시
                info_text = f"Time: {time_display} | Event: {event_type} | IN: {count_in} | OUT: {count_out}"
                cv2.putText(annotated_frame, info_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # 2. 파일 저장
                time_filename = current_real_time.strftime("%H_%M_%S")
                # 파일명에 이벤트 타입 포함 (예: 08_05_12_IN_capture.jpg)
                save_filename = f"{time_filename}_{event_type}_capture.jpg"
                save_path = os.path.join(output_dir, save_filename)
                
                cv2.imwrite(save_path, annotated_frame)
                
                # 3. 로그 기록
                # [시간 | 이벤트타입 | IN누적 | OUT누적 | 파일명]
                log_line = f" [{current_minute_str}]   |  {event_type:^5}  | {count_in:^5} | {count_out:^5} |  {save_filename}\n"
                f.write(log_line)
                f.flush()
                
                print(f"🚨 출입 감지! [{time_display}] {event_type} 발생 -> {save_filename}")
                
                last_logged_minute = current_minute_str

        # 다음 프레임용 저장
        prev_bees = current_bees
        
        if i % 1000 == 0:
            print(f"... {(i/total_frames)*100:.1f}% 진행 중")

cap.release()
print(f"\n완료! 로그 파일: {output_log_path}")