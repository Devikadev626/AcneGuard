import os

dataset_path = r"C:\Users\Edu_K_61\Desktop\Projects\AcneGuard\data\raw\severity_dataset\JPEGImages"

total_images = len(os.listdir(dataset_path))

print("=" * 50)
print("SEVERITY DATASET REVIEW")
print("=" * 50)
print(f"Total Images : {total_images}")