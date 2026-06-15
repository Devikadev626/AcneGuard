from PIL import Image
import os

dataset_path = r"C:\Users\Edu_K_61\Desktop\Projects\AcneGuard\data\raw\severity_dataset\JPEGImages"

sizes = []

for image in os.listdir(dataset_path):

    try:
        img = Image.open(
            os.path.join(dataset_path, image)
        )

        sizes.append(img.size)

    except:
        pass

print("Unique Sizes:", len(set(sizes)))

print("\nSample Sizes:")

for size in list(set(sizes))[:20]:
    print(size)