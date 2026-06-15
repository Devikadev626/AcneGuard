from collections import Counter

file_path = "data/raw/severity_dataset/NNEW_trainval_1.txt"

severity_counter = Counter()

with open(file_path, "r") as f:
    for line in f:
        parts = line.strip().split()

        if len(parts) >= 2:
            severity = parts[1]
            severity_counter[severity] += 1

print("=" * 50)
print("SEVERITY CLASS DISTRIBUTION")
print("=" * 50)

for severity, count in sorted(severity_counter.items()):
    print(f"Grade {severity}: {count} images")