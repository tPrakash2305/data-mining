import os
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder, save_results=True):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.3f}\n")

    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_
    )
    cm = confusion_matrix(y_test, y_pred)

    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    # Save results
    if save_results:
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)

        metrics_path = os.path.join(results_dir, "metrics_report.txt")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(f"Accuracy: {acc:.3f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title("Confusion Matrix - Gaussian Naïve Bayes")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_path = os.path.join(results_dir, "confusion_matrix.png")
        plt.savefig(cm_path)

    return model
