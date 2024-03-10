from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import os
import sys

# Add the 'src' directory to the sys.path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from src.utils import load_data

def train_model(X_train, y_train, X_test, y_test, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32, validation_batch_size=32)


    model.save("/users/rahulnarramneni/Documents/Projects/MartianSurfaceAnamolyDetection/models/trained_models/CNN.h5")

if __name__ == "__main__":
    data_path = "/users/rahulnarramneni/Documents/Projects/MartianSurfaceAnamolyDetection/data/processed"
    X_train, y_train, X_test, y_test = load_data(data_path)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Get the number of unique classes
    num_classes = len(set(y_train))

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    train_model(X_train, y_train, X_test, y_test, num_classes)
