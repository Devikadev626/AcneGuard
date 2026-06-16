import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms

from torch.utils.data import DataLoader

from dataset import AcneSeverityDataset

# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

# ==========================================
# TRANSFORMS
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor()
])

# ==========================================
# DATASET
# ==========================================

dataset = AcneSeverityDataset(
    image_dir="data/raw/severity_dataset/JPEGImages",
    label_file="data/raw/severity_dataset/NNEW_trainval_0.txt",
    transform=train_transform
)

print(
    f"Dataset Size: {len(dataset)}"
)

# ==========================================
# DATALOADER
# ==========================================

train_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

# ==========================================
# MODEL
# ==========================================

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(
        num_features,
        4
    )
)

model = model.to(device)

# ==========================================
# CLASS WEIGHTS
# ==========================================

weights = torch.tensor(
    [1.0, 0.8, 3.5, 4.5],
    dtype=torch.float
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# ==========================================
# TRAINING
# ==========================================

epochs = 20

best_accuracy = 0

for epoch in range(epochs):

    model.train()

    running_loss = 0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    accuracy = (
        100 * correct / total
    )

    avg_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            model.state_dict(),
            "models/best_acne_model.pth"
        )

        print(
            "Best Model Saved"
        )

print("\nTraining Complete")
print(
    f"Best Accuracy: {best_accuracy:.2f}%"
)