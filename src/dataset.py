import os

from PIL import Image

from torch.utils.data import Dataset

class AcneSeverityDataset(Dataset):

    def __init__(self, image_dir, label_file, transform=None):

        self.image_dir = image_dir
        self.transform = transform

        self.samples = []

        with open(label_file, "r") as f:

            for line in f:

                parts = line.strip().split()

                image_name = parts[0]
                severity = int(parts[1])

                self.samples.append(
                    (image_name, severity)
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        image_name, label = self.samples[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label