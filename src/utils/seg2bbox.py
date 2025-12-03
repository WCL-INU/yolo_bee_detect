import os
from pathlib import Path
import dotenv

dotenv.load_dotenv()


def main():

    data_dir = os.getenv("DATA_PATH", "./data")

    test_dir = Path(data_dir) / "test"
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "valid"

    test_label_dir = test_dir / "labels"
    train_label_dir = train_dir / "labels"
    val_label_dir = val_dir / "labels"

    label_dirs = [test_label_dir, train_label_dir, val_label_dir]

    for label_dir in label_dirs:
        for file in label_dir.glob("*.txt"):
            with open(file, "r") as f:
                tmp_labels = [line.strip().split() for line in f.readlines()]
            labels = []
            for label in tmp_labels:
                if len(label) > 5:
                    x_min = min([float(label[i]) for i in range(1, len(label), 2)])
                    y_min = min([float(label[i]) for i in range(2, len(label), 2)])
                    x_max = max([float(label[i]) for i in range(1, len(label), 2)])
                    y_max = max([float(label[i]) for i in range(2, len(label), 2)])
                    xc = (x_min + x_max) / 2
                    yc = (y_min + y_max) / 2
                    bw = x_max - x_min
                    bh = y_max - y_min
                    label = [label[0], str(xc), str(yc), str(bw), str(bh)]
                labels.append(label)
            with open(file, "w") as f:
                for label in labels:
                    f.write(" ".join(label) + "\n")


if __name__ == "__main__":
    main()
