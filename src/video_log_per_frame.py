import cv2
from ultralytics import YOLO
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

CHUNK_SIZE = 2000000  # 2 million rows per chunk


def main():
    # 1. 모델 및 영상 경로 설정
    model_path = os.getenv("YOLO_MODEL_PATH")
    video_path = os.getenv("VIDEO_PATH")
    output_dir = os.getenv("OUTPUT_DIR")

    if not model_path or not video_path or not output_dir:
        print("환경 변수 YOLO_MODEL_PATH, VIDEO_PATH, OUTPUT_DIR를 설정해주세요.")
        sys.exit()

    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_log_path = []
    print("모델 경로:", model_path)
    print("영상 경로:", video_path)
    print("영상 이름:", video_name)
    print("출력 디렉토리:", output_dir)
    print(
        "출력 로그 경로 형태:",
        os.path.join(output_dir, f"{video_name}_<chunk_number>_<frame_index>.parquet"),
    )

    # YOLO 모델 로드
    model = YOLO(model_path)

    # 2. 영상 정보 읽기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("영상을 열 수 없습니다.")
        sys.exit()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 박스 기록 변수
    # df = pd.DataFrame(columns=["frame", "box_index", "x", "y", "width", "height"])
    df = pd.DataFrame(
        {
            "frame": pd.Series(dtype="uint32"),
            "box_index": pd.Series(dtype="uint32"),
            "x": pd.Series(dtype="uint16"),
            "y": pd.Series(dtype="uint16"),
            "width": pd.Series(dtype="uint16"),
            "height": pd.Series(dtype="uint16"),
        }
    )

    # 추론 시작
    results = model.predict(source=video_path, stream=True, verbose=False)
    frames = []
    box_indices = []
    xs = []
    ys = []
    ws = []
    hs = []
    frame_index = -1
    chunk_number = -1
    for _, result in enumerate(results):
        frame_index += 1

        boxes = result.boxes

        # --------------------------------------------------
        # 1. 로그 기록
        # --------------------------------------------------

        if boxes is None or len(boxes) == 0:
            continue

        # 박스 정보 나열
        for box in boxes:
            x_i, y_i, width_i, height_i = box.xywh[0].cpu().numpy()
            xs.append(int(x_i))
            ys.append(int(y_i))
            ws.append(int(width_i))
            hs.append(int(height_i))
        frames.extend([frame_index] * len(boxes))
        box_indices.extend(list(range(len(boxes))))

        # 청크 단위로 저장
        if len(frames) >= CHUNK_SIZE:
            chunk_number = chunk_number + 1

            chunk_data = pd.DataFrame(
                {
                    "frame": pd.Series(frames[:CHUNK_SIZE], dtype="uint32"),
                    "box_index": pd.Series(box_indices[:CHUNK_SIZE], dtype="uint32"),
                    "x": pd.Series(xs[:CHUNK_SIZE], dtype="uint16"),
                    "y": pd.Series(ys[:CHUNK_SIZE], dtype="uint16"),
                    "width": pd.Series(ws[:CHUNK_SIZE], dtype="uint16"),
                    "height": pd.Series(hs[:CHUNK_SIZE], dtype="uint16"),
                }
            )

            frames = frames[CHUNK_SIZE:]
            box_indices = box_indices[CHUNK_SIZE:]
            xs = xs[CHUNK_SIZE:]
            ys = ys[CHUNK_SIZE:]
            ws = ws[CHUNK_SIZE:]
            hs = hs[CHUNK_SIZE:]

            chunk_data.to_parquet(
                os.path.join(
                    output_dir,
                    f"{video_name}_{chunk_number:04d}_{frame_index:04d}.parquet",
                ),
                engine="pyarrow",
                compression="zstd",
                index=False,
            )

        # # 진행률 표시 (1000프레임마다)
        # if frame_index % 1000 == 0:
        progress = (frame_index / total_frames) * 100
        print(f"... {progress:.2f}% 진행 중", end="\r")

    # 남은 데이터 저장
    if len(df) > 0:
        chunk_number = chunk_number + 1

        chunk_data = pd.DataFrame(df[:CHUNK_SIZE])
        df = df[CHUNK_SIZE:]

        chunk_data.to_parquet(
            os.path.join(
                output_dir,
                f"{video_name}_{chunk_number:04d}_{frame_index:04d}.parquet",
            ),
            engine="pyarrow",
            compression="zstd",
            index=False,
        )

    cap.release()
    print(f"생성된 로그 파일: {output_log_path}")
    return


if __name__ == "__main__":
    main()
