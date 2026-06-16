import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Acne Severity Confusion Matrix"
)

plt.show()