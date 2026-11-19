from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
ASSET_DIR = ROOT_DIR / "assets"
CLASSIFIER_PATH = MODEL_DIR / "best_ecocycle_3class_model.keras"
DETECTOR_PATH = MODEL_DIR / "best.pt"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["PET", "HDPEM", "OTHER"]
CLASS_DESCRIPTIONS = {
    "PET": "Polyethylene Terephthalate",
    "HDPEM": "High-Density Polyethylene",
    "OTHER": "Other materials represented in the training dataset",
}
CLASS_GUIDANCE = {
    "PET": "PET detected. This material is commonly used in beverage bottles and can be directed towards PET recovery streams.",
    "HDPEM": "HDPE detected. This material is commonly used in rigid household and personal-care packaging.",
    "OTHER": "The crop does not match the PET or HDPEM classes with the highest confidence. Review the material before sorting.",
}
DETECTOR_LABELS = {
    0: "pet", 1: "pead", 2: "mixed_plastic_soft", 3: "ecal",
    4: "metal", 5: "cardboard", 6: "mixed_plastic_rigid", 7: "pet_oleo",
}
TEST_ACCURACY = "95.85%"
TEST_IMAGE_COUNT = "723"
PALETTE = {
    "forest": "#176344", "forest_dark": "#104B34", "mint": "#EAF4EE",
    "teal": "#16877A", "ink": "#17251E", "muted": "#6B7B70",
    "line": "#E2EAE4", "canvas": "#F5F8F5", "white": "#FFFFFF",
    "amber": "#B7791F", "red": "#B54747",
}
NAV_ITEMS = ["Home", "AI Scanner", "Dashboard", "Impact", "How it works", "Smart Sorting System", "About"]
