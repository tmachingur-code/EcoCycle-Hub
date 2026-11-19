# EcoCycle Energy Hub

EcoCycle Energy Hub is an AI-powered waste material classification prototype built with Streamlit and MobileNetV2. The application analyses uploaded images and predicts whether the material belongs to one of three categories: `PET`, `HDPEM`, or `OTHER`.

The project explores how artificial intelligence can support smarter material identification and contribute to more effective waste recovery and circular economy initiatives.

## Features

- Responsive, light-themed interface with seven pages.
- AI-powered image classification using MobileNetV2.
- Classification of uploaded images into `PET`, `HDPEM`, and `OTHER`.
- Prediction confidence scores and class probability breakdowns.
- Material-specific sorting recommendations.
- Session-level scan dashboard with material distribution statistics.
- Scan history with CSV export.
- Impact and future smart-sorting development information.

## Project Structure

```text
EcoCycle_Energy_Hub/
├── app.py
├── requirements.txt
├── README.md
├── config/
│   └── settings.py
├── core/
│   ├── model_loader.py
│   ├── classifier.py
│   ├── image_processing.py
│   └── history.py
├── components/
│   ├── styles.py
│   └── layout.py
├── pages/
│   ├── home.py
│   ├── scanner.py
│   ├── dashboard.py
│   ├── impact.py
│   ├── how_it_works.py
│   ├── sorting_system.py
│   └── about.py
├── models/
│   └── best_ecocycle_3class_model.keras
└── assets/
```

## Setup and Installation

Use Python 3.10 or 3.11 in a virtual environment.

### 1. Clone the repository

```bash
git clone https://github.com/tmachingur-code/EcoCycle-Hub
cd EcoCycle_Energy_Hub
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access EcoCycle Energy Hub.

## AI Model: MobileNetV2

EcoCycle currently uses MobileNetV2, a convolutional neural network architecture for image classification.

**Model configuration:**

- Architecture: MobileNetV2
- Input image size: 224 × 224 pixels
- Input preprocessing: MobileNetV2 preprocessing
- Model file: `best_ecocycle_3class_model.keras`
- Number of classes: 3

### Classification workflow

1. **Image upload:** The user uploads a JPG, JPEG or PNG image.
2. **Image preprocessing:** The image is converted to RGB and resized to 224 × 224 pixels.
3. **AI classification:** MobileNetV2 analyses the complete uploaded image.
4. **Prediction:** The model produces probabilities for PET, HDPEM and OTHER.
5. **Results:** EcoCycle displays the predicted material, confidence score and probability distribution.
6. **Recommendation:** The application provides guidance based on the predicted material.

## Material Categories

| Class | Description |
|---|---|
| PET | Polyethylene Terephthalate |
| HDPEM | High-Density Polyethylene |
| OTHER | Other material categories represented in the training dataset |

## Model Evaluation

The MobileNetV2 classifier achieved **95.85% accuracy** on its held-out test dataset of **723 images**.

These results describe performance on the evaluation dataset and are not a guarantee of accuracy on every real-world image.

## Smart Sorting: Future Development

The current application focuses on image-based material classification. Future development will explore how AI predictions could support more advanced waste sorting workflows.

Potential development areas include:

- Exploring YOLO11 for future object detection.
- Improving model performance with more representative training data.
- Integrating camera systems into waste sorting workflows.
- Investigating conveyor integration and automated material separation.
- Evaluating the system under real-world waste management conditions.

These are future development goals and are not active features of the current application.

## Important Scope Note

EcoCycle currently performs image-based material classification using MobileNetV2. It analyses the complete uploaded image rather than detecting and classifying individual objects.

Automated object detection, conveyor control, robotic actuation and physical material separation are not implemented in this version.

All predictions should be reviewed by a person before operational sorting decisions are made.

## Troubleshooting

- **Model file not found:** Ensure `best_ecocycle_3class_model.keras` is inside the `models/` directory.
- **TensorFlow installation errors:** Use a compatible Python version and follow the TensorFlow installation guidance for your operating system.
- **Slow first scan:** Initial model loading and inference may take longer. Subsequent requests reuse the cached model.
- **Image upload errors:** Ensure the uploaded file is a valid JPG, JPEG or PNG image.
- **Unexpected predictions:** Use clear, well-lit images and review the confidence score before making sorting decisions.

---

**EcoCycle Energy Hub**

*Turning waste intelligence into circular opportunities.*
