import os
import dotenv

def main():
    dotenv.load_dotenv()
    data_dir = os.getenv("DATA_PATH", "./data")
    print(f"Data directory is set to: {data_dir}")

    train_data_path = os.path.join(data_dir, "train")
    train_images_path = os.path.join(train_data_path, "images")
    train_labels_path = os.path.join(train_data_path, "labels")
    train_images = [f for f in os.listdir(train_images_path) if f.endswith(".jpg")]
    train_labels = [f for f in os.listdir(train_labels_path) if f.endswith(".txt")]

    unmatched_files = []
    for image_file in train_images:
        label_file = image_file.replace(".jpg", ".txt")
        if label_file not in train_labels:
            unmatched_files.append(image_file)

    print("Unmatched image files (no corresponding label file):")
    for file in unmatched_files:
        print(file)


if __name__ == "__main__":
    main()
