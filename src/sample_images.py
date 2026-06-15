import os
import matplotlib.pyplot as plt
from PIL import Image

dataset_path = "data/raw/train"

fig, axes = plt.subplots(1, 5, figsize=(18, 5))

for i, folder in enumerate(sorted(os.listdir(dataset_path))):

    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):

        first_image = os.listdir(folder_path)[0]

        image_path = os.path.join(folder_path, first_image)

        img = Image.open(image_path)

        axes[i].imshow(img)
        axes[i].set_title(folder)
        axes[i].axis("off")

plt.tight_layout()
plt.show()