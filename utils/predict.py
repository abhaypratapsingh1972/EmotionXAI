import torch
import timm

from PIL import Image
from torchvision import transforms

class_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

device = "cpu"

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=7
)

model.load_state_dict(
    torch.load(
        "best_b0_finetuned.pth",
        map_location=device
    )
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

def predict_emotion(image):

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(tensor)

        probs = torch.softmax(output, dim=1)[0]

    top_probs, top_indices = torch.topk(
        probs,
        3
    )

    return (
        class_names[torch.argmax(probs)],
        probs.max().item(),
        top_probs,
        top_indices
    )

def preprocess_image(image):

    tensor = transform(image)

    return tensor.unsqueeze(0)