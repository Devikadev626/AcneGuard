from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import AcneSeverityDataset


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


dataset = AcneSeverityDataset(
    image_dir="data/raw/severity_dataset/JPEGImages",
    label_file="data/raw/severity_dataset/NNEW_trainval_1.txt",
    transform=transform
)


print("Dataset Size:", len(dataset))


loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)


images, labels = next(iter(loader))

print("Image Batch Shape:", images.shape)
print("Labels:", labels)