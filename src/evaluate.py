import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms

from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import pandas as pd

from dataset import AcneSeverityDataset


# ==========================================
# CONFIG
# ==========================================

IMAGE_DIR = (
    "data/raw/severity_dataset/JPEGImages"
)

TEST_FILE = (
    "data/raw/severity_dataset/NNEW_test_1.txt"
)

MODEL_PATH = (
    "models/best_acne_model.pth"
)

BATCH_SIZE = 8

NUM_CLASSES = 4


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")


# ==========================================
# TRANSFORMS
# ==========================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ==========================================
# DATASET
# ==========================================

test_dataset = AcneSeverityDataset(
    image_dir=IMAGE_DIR,
    label_file=TEST_FILE,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(
    f"Test Dataset Size: {len(test_dataset)}"
)


# ==========================================
# MODEL
# ==========================================

model = models.resnet18(
    weights=None
)

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(
        num_features,
        NUM_CLASSES
    )
)

model = model.to(device)


# ==========================================
# LOAD MODEL
# ==========================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print(
    "Model Loaded Successfully"
)


# ==========================================
# EVALUATION
# ==========================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, preds = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            preds.cpu().numpy()
        )


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        digits=4
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix\n")

print(cm)


# ==========================================
# SAVE PREDICTIONS
# ==========================================

results = pd.DataFrame({

    "Actual": all_labels,

    "Predicted": all_predictions

})

results.to_csv(
    "reports/predictions.csv",
    index=False
)

print(
    "\nPredictions Saved Successfully"
)