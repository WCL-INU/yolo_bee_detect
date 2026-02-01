import os
import sys
import time
import selectors
import subprocess
from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Optional


MAX_WORKERS = 3
REFRESH_SEC = 0.0001

YOLO_MODEL_PATH = (
    "/home/siu/projects/yolo_bee_detect/models/train_20251231/weights/best.pt"
)
OUTPUT_DIR = "/home/siu/projects/yolo_bee_detect/tmp/report_dir"


@dataclass
class Job:
    slot: int
    video: str
    p: subprocess.Popen
    last: str = ""
    buf: str = ""
    rc: Optional[int] = None


def spawn_job(slot: int, video_dir: str, video: str) -> Job:
    env = dict(os.environ)
    env["VIDEO_PATH"] = os.path.join(video_dir, video)
    env["YOLO_MODEL_PATH"] = YOLO_MODEL_PATH
    env["OUTPUT_DIR"] = OUTPUT_DIR

    # stdout/stderr를 부모가 수집해서 화면을 재구성 → 섞임 방지
    p = subprocess.Popen(
        ["uv", "run", "./src/video_log_per_frame.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=False,
        bufsize=0,
    )
    return Job(slot=slot, video=video, p=p, last="", buf="")


def render(jobs: List[Optional[Job]]):
    sys.stdout.write("\033[2J\033[H")  # clear + home
    running = sum(1 for j in jobs if j and j.p.poll() is None)
    total = sum(1 for j in jobs if j)
    sys.stdout.write(f"running {running}/{MAX_WORKERS} (active slots {total})\n")
    sys.stdout.write("=" * 100 + "\n")

    for i, j in enumerate(jobs):
        if j is None:
            sys.stdout.write(f"[{i:02d}] (idle)\n")
            continue
        rc = j.p.poll()
        if rc is None:
            st = "RUN"
        else:
            st = "OK" if rc == 0 else f"FAIL({rc})"
        line = (j.last or "").strip()
        sys.stdout.write(f"[{i:02d}] {st:10s} {j.video} | {line}\n")

    sys.stdout.flush()


def main(video_dir: str, video_list: List[str]) -> int:
    queue: Deque[str] = deque(video_list)
    sel = selectors.DefaultSelector()

    # 고정 슬롯(0..MAX_WORKERS-1). 각 슬롯에 Job 또는 None
    slots: List[Optional[Job]] = [None] * MAX_WORKERS

    def start_in_slot(slot: int):
        nonlocal slots
        if not queue:
            return
        video = queue.popleft()
        job = spawn_job(slot, video_dir, video)
        slots[slot] = job
        # selectors는 fileobj 등록 (PIPE)
        assert job.p.stdout is not None
        sel.register(job.p.stdout, selectors.EVENT_READ, data=job)

    # 초기 채우기
    for s in range(MAX_WORKERS):
        if not queue:
            break
        start_in_slot(s)

    last_render = 0.0

    while True:
        # 종료 조건: 큐도 비었고, 실행 중인 슬롯도 없음
        any_running = any(j is not None and j.p.poll() is None for j in slots)
        if (not queue) and (not any_running):
            break

        # 출력 수집
        for key, _ in sel.select(timeout=0.05):
            job: Job = key.data
            f = key.fileobj

            chunk = f.read1(4096) if hasattr(f, "read1") else f.read(4096)
            if not chunk:
                # EOF: 등록 해제 (이미 프로세스가 끝났거나 파이프 닫힘)
                try:
                    sel.unregister(f)
                except Exception:
                    pass
                continue

            text = chunk.decode(errors="replace")

            # \r / \n 처리: 최신 1줄만 유지
            for ch in text:
                if ch == "\r":
                    job.last = job.buf
                    job.buf = ""
                elif ch == "\n":
                    if job.buf:
                        job.last = job.buf
                        job.buf = ""
                else:
                    job.buf += ch
            if job.buf:
                job.last = job.buf

        # 종료된 작업 슬롯 회수 + 다음 작업 투입
        for s, job in enumerate(slots):
            if job is None:
                continue
            rc = job.p.poll()
            if rc is None:
                continue

            # 종료 처리
            job.rc = rc
            # 파이프 등록 해제/닫기
            try:
                if job.p.stdout:
                    try:
                        sel.unregister(job.p.stdout)
                    except Exception:
                        pass
                    job.p.stdout.close()
            except Exception:
                pass

            # 이 슬롯 비우고 다음 작업 시작
            slots[s] = None
            start_in_slot(s)

        # 화면 갱신
        now = time.time()
        if now - last_render >= REFRESH_SEC:
            render(slots)
            last_render = now

    # 마지막 렌더
    render(slots)
    return 0


if __name__ == "__main__":
    video_dir = "/mnt/videos"
    video_list = [
        "cut_ANU-25-summer-14_20260106.mp4",
        "cut_ANU-25-summer-15_20260106.mp4",
        "cut_ANU-25-summer-16_20260106.mp4",
        "cut_ANU-25-summer-17_20260106.mp4",
        "cut_ANU-25-summer-18_20260106.mp4",
        "cut_ANU-25-summer-19_20260101.mp4",
        "cut_ANU-25-summer-19_20260104.mp4",
        "cut_ANU-25-summer-19_20260105.mp4",
        "cut_ANU-25-summer-19_20260106.mp4",
        "cut_ANU-25-summer-19_20260108.mp4",
        "cut_ANU-25-summer-20_20260106.mp4",
        "cut_ANU-25-summer-2_20260106.mp4",
        "cut_ANU-25-summer-3_20260106.mp4",
        "cut_ANU-25-summer-4_20260106.mp4",
    ]
    raise SystemExit(main(video_dir, video_list))
