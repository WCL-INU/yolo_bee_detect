import os
import sys
import cv2
import numpy as np
import multiprocessing as mp
from ultralytics import YOLO
import pyarrow as pa
import pyarrow.parquet as pq


# row-group 목표 크기(바이트). 128MB 권장 (64~256MB 범위에서 조정)
ROWGROUP_TARGET_BYTES = 256 * 1024 * 1024

# 큐에 쌓을 최대 블록 개수(메모리/백프레셔 조절)
QUEUE_MAXSIZE = 16


def _estimate_rows_per_rowgroup():
    # frame:uint32(4) + box_index:uint32(4) + x,y,w,h:uint16*4(8) = 16 bytes/row (압축 전)
    bytes_per_row = 16
    return max(1_000_00, ROWGROUP_TARGET_BYTES // bytes_per_row)  # 최소 100k rows


def parquet_writer_worker(out_parquet: str, schema: pa.Schema, q: mp.Queue):
    writer = pq.ParquetWriter(
        out_parquet,
        schema=schema,
        compression="zstd",
        use_dictionary=True,
        write_statistics=False,
    )

    # 워커 내부 버퍼
    buf_frame = []
    buf_boxi = []
    buf_x = []
    buf_y = []
    buf_w = []
    buf_h = []
    buf_rows = 0

    rows_per_rg = _estimate_rows_per_rowgroup()

    def flush():
        nonlocal buf_rows
        if buf_rows == 0:
            return

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

    try:
        while True:
            item = q.get()
            if item is None:
                break

            frame_idx, arr_xywh = item  # arr_xywh: (N,4) uint16
            n = arr_xywh.shape[0]
            if n == 0:
                continue

            # writer에서만 frame/box_index 생성 (메인 코어 부담 감소)
            buf_frame.append(np.full(n, frame_idx, dtype=np.uint32))
            buf_boxi.append(np.arange(n, dtype=np.uint32))

            # arr_xywh는 [x,y,w,h] 순서
            buf_x.append(arr_xywh[:, 0])
            buf_y.append(arr_xywh[:, 1])
            buf_w.append(arr_xywh[:, 2])
            buf_h.append(arr_xywh[:, 3])
            buf_rows += n

            if buf_rows >= rows_per_rg:
                flush()

        flush()
    finally:
        writer.close()


def main():
    model_path = os.getenv("YOLO_MODEL_PATH")
    video_path = os.getenv("VIDEO_PATH")
    output_dir = os.getenv("OUTPUT_DIR")

    if not model_path or not video_path or not output_dir:
        print("환경 변수 YOLO_MODEL_PATH, VIDEO_PATH, OUTPUT_DIR를 설정해주세요.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_parquet = os.path.join(output_dir, f"{video_name}.parquet")

    print("모델 경로:", model_path)
    print("영상 경로:", video_path)
    print("출력:", out_parquet)

    # 프레임 수(진행률용)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("영상을 열 수 없습니다.")
        sys.exit(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    model = YOLO(model_path)

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

    mp.set_start_method("spawn", force=True)
    q = mp.Queue(maxsize=QUEUE_MAXSIZE)
    writer_p = mp.Process(
        target=parquet_writer_worker,
        args=(out_parquet, schema, q),
        daemon=True,
    )
    writer_p.start()

    frame_index = -1

    try:
        # batch는 8/16/24/32로 실험 권장. (GPU가 놀아도 CPU가 병목이면 배치만으론 안 오름)
        results = model.predict(source=video_path, stream=True, verbose=False, batch=16)

        for result in results:
            frame_index += 1

            if frame_index % 10 == 0 and total_frames > 0:
                if frame_index == 0:
                    start_tick = cv2.getTickCount()
                elapsed = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()
                fps = frame_index / elapsed if elapsed > 0 else 0.0
                progress = (frame_index / total_frames) * 100
                print(
                    f"...{frame_index} / {total_frames} ({progress:.2f}%) FPS: {fps:.2f}",
                    end="\r",
                    flush=True,
                )

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xywh = boxes.xywh
            if xywh is None or xywh.shape[0] == 0:
                continue

            # (중요) 프레임당 1회만 CPU로 이동
            arr = xywh.cpu().numpy()  # float (N,4)
            # 픽셀 단위 보장 목적: 반올림 후 uint16
            arr = np.rint(arr).astype(np.int32)

            # 안전 클립 (해상도 범위 안이면 비용 거의 없음)
            # x,y는 최대 1640/1232, w,h도 비슷한 범위라고 가정
            arr[:, 0] = np.clip(arr[:, 0], 0, 65535)
            arr[:, 1] = np.clip(arr[:, 1], 0, 65535)
            arr[:, 2] = np.clip(arr[:, 2], 0, 65535)
            arr[:, 3] = np.clip(arr[:, 3], 0, 65535)

            arr_u16 = arr.astype(np.uint16, copy=False)

            # 큐에 넣기 (큐가 꽉 차면 여기서 block -> 메모리 폭주 방지)
            q.put((frame_index, arr_u16), block=True)

    finally:
        # writer 종료
        q.put(None)
        writer_p.join()

    print(f"\n완료: {out_parquet}")


if __name__ == "__main__":
    main()
