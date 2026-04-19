from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        self.model.eval()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        self.model.zero_grad()
        target_score = output[:, class_idx]
        target_score.backward()

        gradients = self.gradients[0]          # [C, H, W]
        activations = self.activations[0]      # [C, H, W]

        weights = torch.mean(gradients, dim=(1, 2))  # [C]

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy(), class_idx


def overlay_gradcam_on_image(
    pil_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45
) -> Image.Image:
    image_np = np.array(pil_image.convert("RGB"))

    h, w = image_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)

    return Image.fromarray(overlay)