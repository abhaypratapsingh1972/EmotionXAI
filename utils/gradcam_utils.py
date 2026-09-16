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

    # Target the final convolutional feature block
    target_layers = [model.features[-1]]

    # Get predicted class
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    # Tell Grad-CAM which class to explain
    targets = [
        ClassifierOutputTarget(predicted_class)
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    # Convert PIL image to NumPy
    rgb_img = np.array(
        image.resize((224, 224))
    ).astype(np.float32) / 255.0

    # Generate heatmap overlay
    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return visualization
