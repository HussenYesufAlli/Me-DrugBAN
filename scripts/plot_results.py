import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

# Load predictions from CSV
csv_path = "results/test_predictions.csv"
df = pd.read_csv(csv_path)
labels = df["label"].values
probs = df["probability"].values
preds = df["prediction"].values

# --- 1. ROC Curve ---
fpr, tpr, _ = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

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
plt.savefig("results/test_roc_curve.png")
plt.close()
print("ROC curve saved to results/test_roc_curve.png")

# --- 2. Confusion Matrix ---
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/test_confusion_matrix.png")
plt.close()
print("Confusion matrix saved to results/test_confusion_matrix.png")

# --- 3. Precision-Recall Curve ---
precision, recall, _ = precision_recall_curve(labels, probs)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, color="purple", lw=2, label=f"PR curve (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Me-DrugBAN Test Set")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("results/test_precision_recall_curve.png")
plt.close()
print("Precision-Recall curve saved to results/test_precision_recall_curve.png")

# --- 4. Histogram of Predicted Probabilities ---
plt.figure()
plt.hist(probs[labels==0], bins=30, alpha=0.7, label='True 0', color='blue')
plt.hist(probs[labels==1], bins=30, alpha=0.7, label='True 1', color='orange')
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Histogram of Predicted Probabilities")
plt.legend()
plt.tight_layout()
plt.savefig("results/test_pred_prob_histogram.png")
plt.close()
print("Histogram of predicted probabilities saved to results/test_pred_prob_histogram.png")