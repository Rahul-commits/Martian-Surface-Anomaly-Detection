import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd


def load_data(data_path):
    # Load processed data
    train_df = pd.read_csv(os.path.join(data_path, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test_data.csv"))

    # Load images
    X_train = load_images(train_df["image"], data_path)
    X_test = load_images(test_df["image"], data_path)

    # Load labels
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    return X_train, y_train, X_test, y_test

def load_images(image_names, data_path):
    images = []
    try:
        for img_name in image_names:
            img_path = os.path.join("/Users/rahulnarramneni/Documents/Projects/MartianSurfaceAnamolyDetection/data", "raw", "hirise-map-proj-v3", "map-proj-v3", f"{img_name}")
            with Image.open(img_path) as img:
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized.convert('RGB'))
                images.append(img_array)
    except Exception as e:
        print(f"Error loading images: {e}")

    return np.stack(images)

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    X_train, y_train, X_test, y_test = load_data(data_path)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
