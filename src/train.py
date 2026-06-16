import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import AcneSeverityDataset


# =====================================
# CONFIG
# =====================================

IMAGE_DIR = "data/raw/severity_dataset/JPEGImages"

LABEL_FILE = (
    "data/raw/severity_dataset/NNEW_trainval_1.txt"
)

BATCH_SIZE = 8
NUM_CLASSES = 4
EPOCHS = 5
LEARNING_RATE = 0.001


# =====================================
# TRANSFORMS
# =====================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# =====================================
# DATASET
# =====================================

dataset = AcneSeverityDataset(
    image_dir=IMAGE_DIR,
    label_file=LABEL_FILE,
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(f"Dataset Size: {len(dataset)}")


# =====================================
# CNN MODEL
# =====================================

class AcneCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 28 * 28,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                NUM_CLASSES
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# =====================================
# DEVICE
# =====================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# =====================================
# MODEL
# =====================================

model = AcneCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =====================================
# TRAINING LOOP
# =====================================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in dataloader:

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

    epoch_loss = (
        running_loss /
        len(dataloader)
    )

    accuracy = (
        100 * correct / total
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )


# =====================================
# SAVE MODEL
# =====================================

torch.save(
    model.state_dict(),
    "acne_severity_model.pth"
)

print("\nModel Saved Successfully")