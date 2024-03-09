# Martian Surface Anomaly Detection

## Overview

This project aims to detect anomalies on the Martian surface using image data from the HiRISE (High-Resolution Imaging Science Experiment) camera on board NASA's Mars Reconnaissance Orbiter. Anomaly detection is performed using Convolutional Neural Networks (CNN) to analyze and classify images.

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/MartianSurfaceAnomalyDetection.git
    cd MartianSurfaceAnomalyDetection
    ```

2. Set up a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/Mac
    venv\Scripts\activate  # On Windows
    pip install -r requirements.txt
    ```

3. Run the training script:

    ```bash
    python scripts/train_model.py
    ```

4. Evaluate the model:

    ```bash
    python scripts/evaluation.py
    ```

5. Explore the dashboard:

    ```bash
    python scripts/dashboard.py
    ```

## Dependencies

- Python 3.x
- TensorFlow
- scikit-learn
- pandas
- PIL (Pillow)


