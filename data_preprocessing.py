from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from utils import load_data

def train_model(X_train, y_train, X_val, y_val, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

    # Save the trained model
    model.save("../models/trained_models/your_model_name.h5")  # Replace with an appropriate model name

if __name__ == "__main__":
    data_path = "../data/processed"
    X, y = load_data(data_path)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get the number of unique classes
    num_classes = len(set(y))

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)

    train_model(X_train, y_train, X_val, y_val, num_classes)
