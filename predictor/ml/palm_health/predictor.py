from __future__ import annotations

from pathlib import Path
from timeit import default_timer as timer
from typing import Dict

import torch
from PIL import Image

from .gradcam import GradCAM, overlay_gradcam_on_image
from .model import build_resnet18_model
from .transform import PALM_TRANSFORMS


CLASS_NAMES = ["Anthracnose", "Chimaera", "Healthy Leaves"]

_BASE_DIR = Path(__file__).resolve().parents[3]
_MODEL_PATH = _BASE_DIR / "models" / "Palm_Leaves_ResNet18.pth"

_MODEL = None


def _load_model():
    global _MODEL

    if _MODEL is None:
        model = build_resnet18_model(num_classes=len(CLASS_NAMES))
        state_dict = torch.load(_MODEL_PATH, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        _MODEL = model

    return _MODEL


def predict_palm_leaf(image_path: str) -> Dict:
    model = _load_model()

    image = Image.open(image_path).convert("RGB")
    image_tensor = PALM_TRANSFORMS(image).unsqueeze(0)

    start_time = timer()

    with torch.inference_mode():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probs).item())
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probs[predicted_index])

    probabilities = {
        CLASS_NAMES[i]: float(probs[i])
        for i in range(len(CLASS_NAMES))
    }

    elapsed = round(timer() - start_time, 5)

    # Grad-CAM réel sur la dernière couche convolutive de ResNet18
    target_layer = model.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)

    try:
        cam, class_idx = gradcam.generate(image_tensor, class_idx=predicted_index)
        heatmap_image = overlay_gradcam_on_image(image, cam, alpha=0.45)
    finally:
        gradcam.remove_hooks()

    return {
        "predicted_class": predicted_class,
        "predicted_index": predicted_index,
        "confidence": confidence,
        "prediction_time": elapsed,
        "probabilities": probabilities,
        "heatmap_image": heatmap_image,
    }