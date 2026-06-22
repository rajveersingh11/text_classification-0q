"""
End-to-end ML training pipeline:
  - TF-IDF feature extraction
  - Optional model comparison
  - Hyperparameter tuning via RandomizedSearchCV
  - Stratified K-Fold cross-validation
  - MLflow tracking for every experiment
  - Auto-select best model
"""
try:
    import src.dll_loader
except ImportError:
    try:
        import dll_loader
    except ImportError:
        pass

import os
import warnings
warnings.filterwarnings("ignore")
os.environ["LIGHTGBM_VERBOSITY"] = "-1"
os.environ["PYTHONWARNINGS"] = "ignore"

import os
import json
import time
import yaml
import argparse
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, cross_val_score,
)
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, log_loss,
)

try:
    from .data_preprocessing import TextPreprocessor, load_data
    from .model import build_model, MODEL_REGISTRY
except ImportError:  # Support direct execution: python src/train.py
    from data_preprocessing import TextPreprocessor, load_data
    from model import build_model, MODEL_REGISTRY


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_model(model, X, y, classes: list) -> dict:
    """Compute a comprehensive metric set."""
    y_pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision_weighted": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
    }
    # log_loss requires probability
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            metrics["log_loss"] = float(log_loss(y, proba, labels=list(range(len(classes)))))
        except Exception:
            pass
    return metrics, y_pred


def run_single_model(
    model_name: str,
    param_grid: dict,
    X_train, y_train, X_val, y_val, X_test, y_test,
    preprocessor: TextPreprocessor,
    cfg: dict,
    cv_folds: int,
) -> dict:
    """Train + tune + evaluate a single model. Returns summary dict."""
    print(f"\n{'=' * 70}\nTraining: {model_name}\n{'=' * 70}")

    base_model = build_model(model_name, random_state=cfg["training"]["random_state"])

    n_jobs = 1 if model_name in ("lstm", "embedding") else cfg["training"]["n_jobs"]
    n_iter_val = 3 if model_name in ("lstm", "embedding") else 10
    grid_size = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 1
    n_iter_run = min(n_iter_val, grid_size)

    # Cross-validation on training set (before tuning)
    skf = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=cfg["training"]["random_state"]
    )
    cv_scores = cross_val_score(
        base_model, X_train, y_train,
        cv=skf, scoring=cfg["training"]["tuning_scoring"],
        n_jobs=n_jobs,
    )
    print(f"   CV {cfg['training']['tuning_scoring']}: "
          f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Hyperparameter search on validation set
    t0 = time.time()
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid if param_grid else {},
        n_iter=n_iter_run,
        cv=skf,
        scoring=cfg["training"]["tuning_scoring"],
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
        random_state=cfg["training"]["random_state"],
    )
    search.fit(X_train, y_train)
    tune_time = time.time() - t0
    best_model = search.best_estimator_
    print(f"   Best params: {search.best_params_}")
    print(f"   Tuning time: {tune_time:.1f}s")

    # Evaluate on val + test
    val_metrics, val_pred = evaluate_model(best_model, X_val, y_val, preprocessor.get_classes())
    test_metrics, test_pred = evaluate_model(best_model, X_test, y_test, preprocessor.get_classes())

    # Per-class report
    report = classification_report(
        y_test, test_pred,
        target_names=preprocessor.get_classes(),
        output_dict=True,
    )

    return {
        "model_name": model_name,
        "best_params": search.best_params_,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "tune_time_sec": tune_time,
        "model": best_model,
        "report": report,
    }


def run_training(config: dict):
    cfg = config
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # Load data
    train_df, val_df, test_df = load_data(
        cfg["data"]["train_path"], cfg["data"]["val_path"], cfg["data"]["test_path"]
    )
    print(f"Loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Preprocess
    preprocessor = TextPreprocessor(
        max_features=cfg["preprocessing"]["max_features"],
        ngram_range=tuple(cfg["preprocessing"]["ngram_range"]),
        min_df=cfg["preprocessing"]["min_df"],
        max_df=cfg["preprocessing"]["max_df"],
        sublinear_tf=cfg["preprocessing"]["sublinear_tf"],
    )

    X_train, y_train = preprocessor.fit_transform(train_df["text"], train_df["label"])
    X_val = preprocessor.transform(val_df["text"])
    y_val = preprocessor.encode_labels(val_df["label"])
    X_test = preprocessor.transform(test_df["text"])
    y_test = preprocessor.encode_labels(test_df["label"])

    print(f"TF-IDF features: {preprocessor.get_feature_count():,}")
    print(f"Classes: {preprocessor.get_classes()}")

    os.makedirs("artifacts", exist_ok=True)
    preprocessor.save("artifacts/preprocessor.pkl")

    # Determine models to train
    if cfg["models"].get("compare_models"):
        model_names = cfg["models"]["compare_models"]
    else:
        model_names = [cfg["models"]["active_model"]]

    hyperparams = cfg["models"].get("hyperparameters", {})
    results = []

    for model_name in model_names:
        with mlflow.start_run(run_name=model_name) as run:
            # Log shared params
            mlflow.log_params({
                "tfidf_max_features": cfg["preprocessing"]["max_features"],
                "tfidf_ngram_range": str(cfg["preprocessing"]["ngram_range"]),
                "tfidf_min_df": cfg["preprocessing"]["min_df"],
                "tfidf_max_df": cfg["preprocessing"]["max_df"],
                "cv_folds": cfg["training"]["cross_validation_folds"],
                "n_features": preprocessor.get_feature_count(),
                "train_size": len(train_df),
            })

            try:
                is_seq = getattr(MODEL_REGISTRY.get(model_name), "is_sequence_model", False)
                X_tr_in = train_df["text"] if is_seq else X_train
                X_val_in = val_df["text"] if is_seq else X_val
                X_te_in = test_df["text"] if is_seq else X_test

                res = run_single_model(
                    model_name=model_name,
                    param_grid=hyperparams.get(model_name, {}),
                    X_train=X_tr_in, y_train=y_train,
                    X_val=X_val_in, y_val=y_val,
                    X_test=X_te_in, y_test=y_test,
                    preprocessor=preprocessor,
                    cfg=cfg,
                    cv_folds=cfg["training"]["cross_validation_folds"],
                )
                results.append(res)

                # Log best params + metrics
                mlflow.log_params({f"best__{k}": v for k, v in res["best_params"].items()})
                mlflow.log_metrics({
                    "cv_mean": res["cv_mean"],
                    "cv_std": res["cv_std"],
                    "tune_time_sec": res["tune_time_sec"],
                    **{f"val_{k}": v for k, v in res["val_metrics"].items()},
                    **{f"test_{k}": v for k, v in res["test_metrics"].items()},
                })

                # Save report & model
                with open(f"artifacts/{model_name}_report.json", "w") as f:
                    json.dump(res["report"], f, indent=2)
                mlflow.log_artifact(f"artifacts/{model_name}_report.json")

                if cfg["training"].get("select_best"):
                    joblib.dump(res["model"], f"artifacts/{model_name}.joblib")
                    mlflow.sklearn.log_model(res["model"], artifact_path=f"model_{model_name}")

            except Exception as e:
                print(f"ERROR: {model_name} failed: {e}")
                mlflow.log_param("error", str(e))

    # ──────── Select best model ────────
    if not results:
        raise RuntimeError("No models trained successfully.")

    summary = []
    for r in results:
        summary.append({
            "model": r["model_name"],
            "cv_f1_weighted": r["cv_mean"],
            "test_f1_weighted": r["test_metrics"]["f1_weighted"],
            "test_accuracy": r["test_metrics"]["accuracy"],
        })
    summary_df = pd.DataFrame(summary).sort_values("test_f1_weighted", ascending=False)
    print("\nModel Comparison:")
    print(summary_df.to_string(index=False))

    best = summary_df.iloc[0].to_dict()
    best_model_name = best["model"]
    best_result = next(r for r in results if r["model_name"] == best_model_name)

    print(f"\nBest model: {best_model_name} (test F1={best['test_f1_weighted']:.4f})")

    # Persist best model as the canonical artifact
    joblib.dump(best_result["model"], "artifacts/best_model.joblib")
    with open("artifacts/best_model_meta.json", "w") as f:
        json.dump({
            "model_name": best_model_name,
            "best_params": best_result["best_params"],
            "test_metrics": best_result["test_metrics"],
            "val_metrics": best_result["val_metrics"],
        }, f, indent=2)

    return {
        "best_model": best_model_name,
        "summary": summary,
        "best_metrics": best_result["test_metrics"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    result = run_training(load_config(args.config))
    print(f"\nTraining complete. Best model: {result['best_model']}")
