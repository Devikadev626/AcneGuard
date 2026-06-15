# src/dataset_review.py

import os

dataset_path = "data/raw/train"

print("=" * 50)
print("ACNE DATASET REVIEW")
print("=" * 50)

total_images = 0

for folder in sorted(os.listdir(dataset_path)):

    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):

        image_count = len(os.listdir(folder_path))

        total_images += image_count

        print(f"{folder:<15} : {image_count} images")

print("-" * 50)
print(f"Total Images   : {total_images}")
print("=" * 50)