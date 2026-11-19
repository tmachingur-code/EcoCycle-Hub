# EcoCycle Energy Hub

A responsive Streamlit prototype that combines YOLO11 object detection with MobileNetV2 material classification.

## Features

- Responsive, light-themed interface with seven pages.
- YOLO11 detects objects in an uploaded image and draws bounding boxes.
- MobileNetV2 classifies each detected crop into `PET`, `HDPEM`, or `OTHER`.
- Per-object confidence scores and class probabilities.
- Session-level scan dashboard with CSV export.

## Project structure

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
│   ├── detector.py
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
│   ├── best.pt
│   └── best_ecocycle_3class_model.keras
└── assets/
```

## Setup

Use Python 3.10 or 3.11 in a virtual environment. TensorFlow and Ultralytics are large packages; install them in an environment with sufficient disk space and memory.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The two model files must remain in `models/` with the exact filenames shown above. Models are loaded lazily and cached after the first request.

## Model responsibilities

1. YOLO11n detects objects and provides bounding boxes. Its eight detector labels are retained as a separate label system.
2. Each detected region is cropped from the original image.
3. MobileNetV2 classifies the crop into `PET`, `HDPEM`, or `OTHER`.

The reported 95.85% accuracy belongs to the MobileNetV2 held-out test evaluation (723 images), not the combined detection/classification pipeline.

## Important scope note

This application performs image-based object detection and material classification only. It does **not** control a conveyor, robotic arm, actuator, or physical sorting mechanism. Predictions should be reviewed by a person before operational use.

## Troubleshooting

- **Model file not found:** check both files are in `models/` and the names match exactly.
- **TensorFlow install errors:** use Python 3.10/3.11 and follow the TensorFlow installation guidance for your operating system and hardware.
- **Slow first scan:** model loading and first-time inference can take longer; subsequent requests reuse cached models.
- **No detections:** try a clear, well-lit image with objects visible and adjust the detector confidence threshold in the scanner.
