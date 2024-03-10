from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np
import os
import sys

# Add the 'src' directory to the sys.path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from src.utils import load_data

def evaluate_model(model_path, X_test, y_test):
    model = load_model(model_path)
    y_pred = model.predict(X_test)
    
    # Convert predicted probabilities to class labels
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # Print classification report
    print(classification_report(y_test, y_pred_labels))

if __name__ == "__main__":
    data_path = "/Users/rahulnarramneni/Documents/Projects/MartianSurfaceAnamolyDetection/data/processed"
    model_path = "/Users/rahulnarramneni/Documents/Projects/MartianSurfaceAnamolyDetection/models/trained_models/CNN.h5"  # Replace with the actual model name

    _,_,X_test, y_test = load_data(data_path)

    # Load the trained model and evaluate
    evaluate_model(model_path, X_test, y_test)
