import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms

from PIL import Image


# ====================================
# CONFIG
# ====================================

MODEL_PATH = "models/best_acne_model.pth"

CLASS_NAMES = {
    0: "Grade 0",
    1: "Grade 1",
    2: "Grade 2",
    3: "Grade 3"
}


# ====================================
# DEVICE
# ====================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ====================================
# MODEL
# ====================================

model = models.resnet18(
    weights=None
)

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(
        num_features,
        4
    )
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()


# ====================================
# TRANSFORM
# ====================================

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor()
])


# ====================================
# PREDICTION FUNCTION
# ====================================

def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    severity = CLASS_NAMES[
        predicted.item()
    ]

    confidence = round(
        confidence.item() * 100,
        2
    )

    return severity, confidence