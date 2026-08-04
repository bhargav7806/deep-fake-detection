# DeepFake Face Detection using ResNet50

A Deep Learning-based web application that detects whether an image is **Real** or **AI-Generated (DeepFake)** using **Transfer Learning with ResNet50**.

---

## Features

- Binary classification (Real / AI-Generated)
- Transfer Learning using ResNet50
- TensorFlow & Keras implementation
- Image preprocessing pipeline
- Confidence score prediction
- Flask web application
- Easy-to-use interface

---

## Tech Stack

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Flask
- HTML/CSS
- Git

---

## Model Architecture

- Backbone: **ResNet50 (ImageNet Weights)**
- Input Shape: **256 × 256 × 3**
- Transfer Learning + Fine-Tuning
- Binary Classification

---

## Project Structure

```text
deep-fake-detection/
│
├── app.py
├── utils.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── fine_tuned_model.keras
│   └── model.ipynb
│
├── static/
│
├── templates/
│
├── data/
│   ├── Data/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── image_label.csv
│
└── uploads/
```

---

## Dataset

The training dataset is **not included** in this repository because of its large size.

Download or prepare the dataset and place it in the following directory structure:

```text
data/
└── Data/
    ├── train/
    ├── val/
    └── test/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/bhargav7806/deep-fake-detection.git
```

Move into the project directory

```bash
cd deep-fake-detection
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python app.py
```

Open your browser and visit

```text
http://127.0.0.1:5000
```

---

## Prediction

Upload an image and the model predicts:

- ✅ Real
- ❌ AI Generated (DeepFake)

along with the confidence score.

---

## Training

The model was trained using:

- Transfer Learning
- Fine-Tuning
- Data Augmentation
- Binary Classification

---

## Future Improvements

- Vision Transformer (ViT)
- YOLO-based Face Detection
- Explainability using Grad-CAM
- ONNX Conversion
- TensorFlow Lite Deployment
- FastAPI Deployment

---

## License

This project is intended for educational and research purposes.

---

## Author

**Bhargav Jadav**

GitHub: https://github.com/bhargav7806
