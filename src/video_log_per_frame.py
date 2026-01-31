import cv2
from ultralytics import YOLO
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK_SIZE = 200000000  # 200 million rows per chunk


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
    out_parquet = os.path.join(output_dir, f"{video_name}.parquet")

    print("모델 경로:", model_path)
    print("영상 경로:", video_path)
    print("영상 이름:", video_name)
    print("출력 디렉토리:", output_dir)
    print(
        "출력 로그 경로 형태:",
        out_parquet,
    )

    # YOLO 모델 로드
    model = YOLO(model_path)

    # 2. 영상 정보 읽기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("영상을 열 수 없습니다.")
        sys.exit()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 추론 시작
    results = model.predict(source=video_path, stream=True, verbose=False, batch=16)

    # 누적 버퍼 (numpy로 누적, pandas 금지)
    buf_frame = []
    buf_boxi = []
    buf_x = []
    buf_y = []
    buf_w = []
    buf_h = []
    buf_rows = 0

    schema = pa.schema(
        [
            ("frame", pa.uint32()),
            ("box_index", pa.uint32()),
            ("x", pa.uint16()),
            ("y", pa.uint16()),
            ("width", pa.uint16()),
            ("height", pa.uint16()),
        ]
    )

    writer = pq.ParquetWriter(
        out_parquet,
        schema=schema,
        compression="zstd",
        use_dictionary=True,
        write_statistics=False,  # 속도 우선이면 False가 유리한 경우 많음
    )

    frame_index = -1
    try:
        for result in results:
            frame_index += 1

            # # 진행률 표시 (10프레임마다)
            if frame_index % 10 == 0:
                progress = (frame_index / total_frames) * 100
                print(
                    f"...{frame_index} / {total_frames} ({progress:.2f}%) 진행 중",
                    end="\r",
                    flush=True,
                )

            # --------------------------------------------------
            # 1. 로그 기록
            # --------------------------------------------------
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xywh = boxes.xywh
            if xywh is None or xywh.shape[0] == 0:
                continue

            arr = xywh.cpu().numpy()  # (N,4) 한번만
            n = arr.shape[0]

            # uint16 범위 넘어갈 수 있으면 clip 또는 uint32로 바꾸세요.
            x = arr[:, 0].astype(np.uint16)
            y = arr[:, 1].astype(np.uint16)
            w = arr[:, 2].astype(np.uint16)
            h = arr[:, 3].astype(np.uint16)

            buf_frame.append(np.full(n, frame_index, dtype=np.uint32))
            buf_boxi.append(np.arange(n, dtype=np.uint32))
            buf_x.append(x)
            buf_y.append(y)
            buf_w.append(w)
            buf_h.append(h)
            buf_rows += n

            # 청크 단위로 저장
            if buf_rows >= CHUNK_SIZE:
                frame_col = np.concatenate(buf_frame)
                boxi_col = np.concatenate(buf_boxi)
                x_col = np.concatenate(buf_x)
                y_col = np.concatenate(buf_y)
                w_col = np.concatenate(buf_w)
                h_col = np.concatenate(buf_h)

                table = pa.Table.from_arrays(
                    [
                        pa.array(frame_col, type=pa.uint32()),
                        pa.array(boxi_col, type=pa.uint32()),
                        pa.array(x_col, type=pa.uint16()),
                        pa.array(y_col, type=pa.uint16()),
                        pa.array(w_col, type=pa.uint16()),
                        pa.array(h_col, type=pa.uint16()),
                    ],
                    schema=schema,
                )
                writer.write_table(table)

                buf_frame.clear()
                buf_boxi.clear()
                buf_x.clear()
                buf_y.clear()
                buf_w.clear()
                buf_h.clear()
                buf_rows = 0

        # flush 남은 것
        if buf_rows > 0:
            frame_col = np.concatenate(buf_frame)
            boxi_col = np.concatenate(buf_boxi)
            x_col = np.concatenate(buf_x)
            y_col = np.concatenate(buf_y)
            w_col = np.concatenate(buf_w)
            h_col = np.concatenate(buf_h)

            table = pa.Table.from_arrays(
                [
                    pa.array(frame_col, type=pa.uint32()),
                    pa.array(boxi_col, type=pa.uint32()),
                    pa.array(x_col, type=pa.uint16()),
                    pa.array(y_col, type=pa.uint16()),
                    pa.array(w_col, type=pa.uint16()),
                    pa.array(h_col, type=pa.uint16()),
                ],
                schema=schema,
            )
            writer.write_table(table)
    finally:
        writer.close()

    print(f"생성된 로그 파일: {out_parquet}\n")
    return


if __name__ == "__main__":
    main()
