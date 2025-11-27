import cv2
import glob
import numpy as np
from pathlib import Path
import dotenv
import os

dotenv.load_dotenv()

DATA_DIR = os.getenv("DATA_PATH", "./data")
IMAGE_DIR = f"{DATA_DIR}/train/images"
LABEL_DIR = f"{DATA_DIR}/train/labels"

bbox_rel_sizes = []
object_counts = []
positions = []
data_cnt = 0
empty_label_count = 0

for img_path in glob.glob(f"{IMAGE_DIR}/*.jpg"):
    txt_path = Path(LABEL_DIR) / (Path(img_path).stem + ".txt")
    if not txt_path.exists():
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(txt_path, "r") as f:
        labels = [line.strip().split() for line in f.readlines()]

    object_counts.append(len(labels))
    data_cnt += 1
    if len(labels) == 0:
        empty_label_count += 1
        continue

    for lab in labels:
        _, xc, yc, bw, bh = map(float, lab)
        # 상대 크기 저장
        bbox_rel_sizes.append(bw * bh)

        # 중심 좌표 저장
        positions.append((xc, yc))

# 결과 요약
bbox_rel_sizes = np.array(bbox_rel_sizes)
object_counts = np.array(object_counts)
positions = np.array(positions)

print("\n===== Object Size Stats =====")
print(f"mean bbox area ratio : {bbox_rel_sizes.mean():.6f}")
print(f"median area ratio    : {np.median(bbox_rel_sizes):.6f}")
print(f"min / max            : {bbox_rel_sizes.min():.6f} / {bbox_rel_sizes.max():.6f}")

print("\n===== Object Count per Image =====")
print(f"mean objects/image   : {object_counts.mean():.2f}")
print(f"median               : {np.median(object_counts):.2f}")
print(f"min / max            : {object_counts.min()} / {object_counts.max()}")

print("\n===== Positional Heatmap Hints =====")
print(f"mean center (x, y)   : {positions.mean(axis=0)}")

print("\n===== Dataset Info =====")
print(f"total images             : {data_cnt}")
print(f"images with no objects   : {empty_label_count} ({(empty_label_count / data_cnt) * 100:.2f}%)")

import matplotlib.pyplot as plt

plt.hist(bbox_rel_sizes, bins=30)
plt.title("BBox area ratio distribution")
plt.show()

xs, ys = positions[:, 0], positions[:, 1]
plt.hexbin(xs, ys, gridsize=25, cmap="inferno")
plt.title("Object position heatmap")
plt.gca().invert_yaxis()
plt.show()
