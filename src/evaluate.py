try:
    import src.dll_loader
except ImportError:
    try:
        import dll_loader
    except ImportError:
        pass

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, cohen_kappa_score, matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize

try:
    from .data_preprocessing import TextPreprocessor
except ImportError:  # Support direct execution: python src/evaluate.py
    from data_preprocessing import TextPreprocessor


def plot_confusion_matrix(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                cbar_kws={"label": "Normalized frequency"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (counts, normalized by row)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_top_features(model, vectorizer, classes, top_n=15, out_path="artifacts/top_features.png"):
    """For linear models with .coef_, plot top features per class."""
    if not hasattr(model, "coef_"):
        print("   (Skipped top features — model has no coef_)")
        return

    feature_names = np.array(vectorizer.get_feature_names_out())
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 6), sharey=True)
    if len(classes) == 1:
        axes = [axes]

    for i, cls in enumerate(classes):
        top_idx = np.argsort(model.coef_[i])[-top_n:][::-1]
        top_features = feature_names[top_idx]
        top_weights = model.coef_[i][top_idx]
        ax = axes[i]
        colors = ["#1f77b4" if w > 0 else "#d62728" for w in top_weights]
        ax.barh(range(len(top_features)), top_weights, color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_title(cls)
        ax.set_xlabel("Weight")

    plt.suptitle(f"Top {top_n} Features per Class", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   Saved top features plot to {out_path}")


def generate_report(
    model_path: str,
    preprocessor_path: str,
    test_csv: str,
    out_dir: str = "artifacts",
):
    os.makedirs(out_dir, exist_ok=True)

    pre = TextPreprocessor.load(preprocessor_path)
    model = joblib.load(model_path)

    df = pd.read_csv(test_csv)
    y_true = pre.encode_labels(df["label"])
    classes = pre.get_classes()

    # Probabilities if available
    has_proba = hasattr(model, "predict_proba")

    if hasattr(model, "is_sequence_model") and model.is_sequence_model:
        y_pred = model.predict(df["text"])
        y_prob = model.predict_proba(df["text"]) if has_proba else None
    else:
        X = pre.transform(df["text"])
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X) if has_proba else None

    report = {
        "model_class": type(model).__name__,
        "classification_report": classification_report(
            y_true, y_pred, target_names=classes, output_dict=True
        ),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_prob is not None:
        y_bin = label_binarize(y_true, classes=list(range(len(classes))))
        report["macro_roc_auc"] = float(
            roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
        )
        report["per_class_roc_auc"] = {
            cls: float(roc_auc_score(y_bin[:, i], y_prob[:, i]))
            for i, cls in enumerate(classes)
        }

    # Save report
    report_path = os.path.join(out_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    plot_confusion_matrix(y_true, y_pred, classes,
                          out_path=os.path.join(out_dir, "confusion_matrix.png"))

    if pre.vectorizer:
        plot_top_features(
            model, pre.vectorizer, classes,
            top_n=15, out_path=os.path.join(out_dir, "top_features.png")
        )

    # Pretty-print
    cr = report["classification_report"]
    print("\n" + "=" * 78)
    print(f"{'Model:':<20}{type(model).__name__}")
    print("=" * 78)
    print(f"{'Class':<22}{'Precision':>12}{'Recall':>10}{'F1':>10}{'Support':>10}")
    print("-" * 78)
    for c in classes:
        row = cr[c]
        print(f"{c:<22}{row['precision']:>12.3f}{row['recall']:>10.3f}"
              f"{row['f1-score']:>10.3f}{int(row['support']):>10}")
    print("-" * 78)
    print(f"{'Accuracy':<22}{cr['accuracy']:>32.3f}{int(cr['macro avg']['support']):>10}")
    print(f"{'Macro avg':<22}{cr['macro avg']['precision']:>12.3f}"
          f"{cr['macro avg']['recall']:>10.3f}{cr['macro avg']['f1-score']:>10.3f}"
          f"{int(cr['macro avg']['support']):>10}")
    print(f"{'Weighted avg':<22}{cr['weighted avg']['precision']:>12.3f}"
          f"{cr['weighted avg']['recall']:>10.3f}{cr['weighted avg']['f1-score']:>10.3f}"
          f"{int(cr['weighted avg']['support']):>10}")
    print("=" * 78)
    print(f"Cohen's Kappa:     {report['cohen_kappa']:.4f}")
    print(f"Matthews CorrCoef: {report['matthews_corrcoef']:.4f}")
    if "macro_roc_auc" in report:
        print(f"Macro ROC-AUC:     {report['macro_roc_auc']:.4f}")
        for c, auc in report["per_class_roc_auc"].items():
            print(f"  {c:<22}ROC-AUC: {auc:.4f}")
    print(f"\nReport saved to {report_path}")
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/best_model.joblib")
    p.add_argument("--prep", default="artifacts/preprocessor.pkl")
    p.add_argument("--test", default="data/test.csv")
    p.add_argument("--out", default="artifacts")
    args = p.parse_args()
    generate_report(args.model, args.prep, args.test, args.out)
