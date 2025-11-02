import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load predictions from CSV
csv_path = "results/test_predictions.csv"
df = pd.read_csv(csv_path)

labels = df["label"].values
probs = df["probability"].values

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Me-DrugBAN Test Set")
plt.legend(loc="lower right")
plt.tight_layout()

roc_fig_path = "results/test_roc_curve.png"
plt.savefig(roc_fig_path)
plt.close()
print(f"ROC curve saved to {roc_fig_path}")