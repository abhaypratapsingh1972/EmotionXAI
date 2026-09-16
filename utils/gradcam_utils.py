import torch
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_gradcam(
    model,
    input_tensor,
    image
):

    model.eval()

    # EfficientNet-B0 target layer
    target_layers = [model.conv_head]

    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)

        # Handle models that return a tensor
        if isinstance(output, torch.Tensor):
            predicted_class = output.argmax(dim=1).item()
        else:
            predicted_class = output.logits.argmax(dim=1).item()

    # Explain the predicted emotion
    targets = [
        ClassifierOutputTarget(predicted_class)
    ]

    # Create Grad-CAM
    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    # Generate CAM
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    # Convert image to RGB NumPy array
    rgb_img = np.array(
        image.resize((224, 224))
    ).astype(np.float32) / 255.0

    # Create heatmap overlay
    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return visualization
