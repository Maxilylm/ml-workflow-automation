"""Train MNIST classifier using sklearn (faster for demo)."""

import sys
sys.path.insert(0, ".")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
from pathlib import Path


def main():
    print("Loading data...")
    images = np.load("data/raw/mnist_images.npy")
    labels = np.load("data/raw/mnist_labels.npy")

    # Normalize
    X = images / 255.0
    y = labels

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Train Logistic Regression (baseline)
    print("\n--- Training Logistic Regression (baseline) ---")
    lr_model = LogisticRegression(
        max_iter=100,
        solver='saga',
        multi_class='multinomial',
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    lr_model.fit(X_train, y_train)

    # Evaluate
    y_pred_lr = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, y_pred_lr)
    print(f"\nLogistic Regression Accuracy: {lr_acc:.4f}")

    # Train Random Forest
    print("\n--- Training Random Forest ---")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    rf_model.fit(X_train, y_train)

    # Evaluate
    y_pred_rf = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    print(f"\nRandom Forest Accuracy: {rf_acc:.4f}")

    # Pick best model
    if rf_acc > lr_acc:
        best_model = rf_model
        best_name = "RandomForest"
        best_acc = rf_acc
        y_pred = y_pred_rf
    else:
        best_model = lr_model
        best_name = "LogisticRegression"
        best_acc = lr_acc
        y_pred = y_pred_lr

    print(f"\n=== Best Model: {best_name} with accuracy {best_acc:.4f} ===")

    # Generate detailed metrics
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Save model
    Path("models").mkdir(exist_ok=True)
    model_path = "models/mnist_sklearn.joblib"
    joblib.dump({
        "model": best_model,
        "model_type": best_name,
        "accuracy": best_acc,
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # Save metrics
    Path("reports/figures/mnist").mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": float(best_acc),
        "precision_macro": float(report['macro avg']['precision']),
        "recall_macro": float(report['macro avg']['recall']),
        "f1_macro": float(report['macro avg']['f1-score']),
        "precision_weighted": float(report['weighted avg']['precision']),
        "recall_weighted": float(report['weighted avg']['recall']),
        "f1_weighted": float(report['weighted avg']['f1-score']),
        "per_class": {
            "precision": [float(report[str(i)]['precision']) for i in range(10)],
            "recall": [float(report[str(i)]['recall']) for i in range(10)],
            "f1": [float(report[str(i)]['f1-score']) for i in range(10)],
            "support": [int(report[str(i)]['support']) for i in range(10)],
        }
    }

    with open("reports/figures/mnist/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to reports/figures/mnist/metrics.json")

    # Save confusion matrix
    np.save("reports/figures/mnist/confusion_matrix.npy", cm)

    # Create confusion matrix plot
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {best_name}')
    plt.tight_layout()
    plt.savefig("reports/figures/mnist/confusion_matrix.png", dpi=150)
    print("Confusion matrix saved to reports/figures/mnist/confusion_matrix.png")

    # Feature importance for RF
    if best_name == "RandomForest":
        importance = best_model.feature_importances_.reshape(28, 28)
        plt.figure(figsize=(8, 8))
        plt.imshow(importance, cmap='hot')
        plt.colorbar()
        plt.title('Feature Importance (Pixel Importance)')
        plt.tight_layout()
        plt.savefig("reports/figures/mnist/feature_importance.png", dpi=150)
        print("Feature importance saved to reports/figures/mnist/feature_importance.png")

    return best_acc


if __name__ == "__main__":
    main()
