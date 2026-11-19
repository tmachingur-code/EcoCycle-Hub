from PIL import Image
from core.classifier import classify_crop

def detect_and_classify(image: Image.Image, detector, classifier, confidence_threshold=0.25):
    """Detect objects with YOLO, then classify each detected crop with MobileNetV2."""
    source = image.convert("RGB")
    predictions = detector.predict(source, imgsz=640, conf=confidence_threshold, verbose=False)
    if not predictions:
        return source, []
    result = predictions[0]
    annotated = Image.fromarray(result.plot()[:, :, ::-1])
    detections = []
    if result.boxes is None:
        return annotated, detections
    width, height = source.size
    for idx, box in enumerate(result.boxes):
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = source.crop((x1, y1, x2, y2))
        class_result = classify_crop(crop, classifier)
        detector_id = int(box.cls[0])
        detector_name = detector.names.get(detector_id, str(detector_id))
        detections.append({
            "object": idx + 1,
            "detector_label": str(detector_name),
            "detector_confidence": float(box.conf[0]),
            "material": class_result["material"],
            "confidence": class_result["confidence"],
            "probabilities": class_result["probabilities"],
            "box": [x1, y1, x2, y2],
            "crop": crop,
        })
    return annotated, detections
