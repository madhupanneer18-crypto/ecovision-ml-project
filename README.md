# ♻️ EcoVision AI — Smart Waste Classification and Detection

## Project Overview
Automated Real-Time Waste Classification System Using Deep Learning.

EcoVision AI is a Python and Streamlit application that uses YOLOv8 deep learning models to classify and detect waste in images.

## Features
- Whole-image waste classification
- Multiple-object waste detection with bounding boxes
- Camera photo analysis
- Interactive analytics dashboard
- Prediction history and CSV export
- Recycling and disposal guidance

## Waste Categories
Classification: Plastic, Glass, Metal, Paper, Cardboard, Trash.

Detection: Glass, Metal, Paper, Plastic, Residual.

## Model Performance
- Classification validation accuracy: 91.3%
- Detection precision at epoch 10: 78.1%
- Detection recall at epoch 10: 79.1%
- Detection mAP@50 at epoch 10: 81.6%
- Detection mAP@50-95 at epoch 10: 59.4%

## Technologies
Python, Streamlit, YOLOv8, PyTorch, OpenCV, Pandas, Plotly, Pillow.

## Installation
Clone the repository and install dependencies:

    git clone https://github.com/madhupanneer18-crypto/ecovision-ml-project.git
    cd ecovision-ml-project
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Run the Dashboard

    streamlit run app.py

## Model Weights
Trained model weights are not included in this repository.

Expected classification model path:

    runs/classify/train/weights/best.pt

Expected detection model path:

    runs/detect/runs/detect/waste_v1-2/weights/best.pt

Place the trained weights in these locations before running predictions.

## Limitations
Camera analysis uses still images rather than continuous live video. Predictions may be incorrect, and recycling rules vary by location.
