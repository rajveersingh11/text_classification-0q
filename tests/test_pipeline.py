"""Unit tests for the ML pipeline."""

import sys, os
sys.path.insert(0, os.path.abspath("."))

try:
    import src.dll_loader
except ImportError:
    pass

import pandas as pd

import numpy as np
from src.data_preprocessing import TextPreprocessor
from src.model import build_model, MODEL_REGISTRY


def test_clean_text():
    out = TextPreprocessor.clean_text("Payment $129.00 for order #45213!! http://x.com")
    assert "AMOUNT" in out
    assert "ORDERID" in out
    assert "URL" in out
    print("PASS: test_clean_text")


def test_fit_transform_sparse():
    p = TextPreprocessor(max_features=500, ngram_range=(1, 1), min_df=1)
    texts = pd.Series([
        "payment failed", "app crashes", "refund please",
        "delivery delayed", "product question",
    ])
    labels = pd.Series([
        "Payment Issue", "Technical Problem", "Refund Request",
        "Delivery Issue", "Product Inquiry",
    ])
    X, y = p.fit_transform(texts, labels)
    assert X.shape[0] == 5
    assert len(y) == 5
    assert p.get_feature_count() > 0
    print("PASS: test_fit_transform_sparse")


def test_all_models_build():
    for name in MODEL_REGISTRY:
        m = build_model(name)
        assert m is not None
    print(f"PASS: test_all_models_build ({len(MODEL_REGISTRY)} models)")


def test_model_train_predict():
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=100, n_features=20, n_classes=5,
        n_informative=10, random_state=42,
    )
    X = np.abs(X)  # Match the non-negative TF-IDF features used in production.
    for name in ["naive_bayes"]:
        m = build_model(name)
        m.fit(X, y)
        preds = m.predict(X)
        assert len(preds) == 100
    print("PASS: test_model_train_predict")


if __name__ == "__main__":
    test_clean_text()
    test_fit_transform_sparse()
    test_all_models_build()
    test_model_train_predict()
    print("\nAll tests passed")
