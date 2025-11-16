import os
import cv2
import dotenv
from pathlib import Path

dotenv.load_dotenv()
model_path = os.getenv("MODEL_PATH", "./model.pth")
input_data_path = os.getenv("INPUT_DATA_PATH", "./input_data")
output_data_path = os.getenv("OUTPUT_DATA_PATH", "./output_data")

import torch

model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def main():

    f = plt.figure(1)

    input_files = [f for f in os.listdir(input_data_path) if f.endswith(".jpg")]
    input_files.sort()
    # Path(output_data_path).mkdir(parents=True, exist_ok=True)

    for file_name in input_files:
        input_file_path = os.path.join(input_data_path, file_name)

        image = cv2.imread(input_file_path)
        results = model(image)
        results.print()
        results.show()
        
        # cv2.imshow("Input Image", image)
        print(f"Processing file: {file_name}")
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.show(block=True)
        # f.clf()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.pause(1)


if __name__ == "__main__":
    main()
