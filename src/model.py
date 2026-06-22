from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from typing import Dict, Any

try:
    from .lstm_model import LSTMClassifier
except ImportError:
    from lstm_model import LSTMClassifier

try:
    from .embedding_model import EmbeddingClassifier
except ImportError:
    from embedding_model import EmbeddingClassifier


MODEL_REGISTRY: Dict[str, type] = {
    "naive_bayes": MultinomialNB,
    "xgboost": XGBClassifier,
    "lstm": LSTMClassifier,
    "embedding": EmbeddingClassifier,
}


def build_model(name: str, random_state: int = 42, **kwargs) -> BaseEstimator:
    """
    Build a model by name. Returns an *unfitted* sklearn-compatible estimator.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    factory = MODEL_REGISTRY[name]
    # MultinomialNB does not support random_state parameter, so we exclude it.
    if name == "naive_bayes":
        return factory(**kwargs)
    return factory(random_state=random_state, **kwargs)


def get_default_param_grid(name: str) -> Dict[str, list]:
    """Conservative default grid if user doesn't supply one."""
    defaults = {
        "naive_bayes": {
            "alpha": [0.1, 0.5, 1.0],
        },
        "xgboost": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        },
        "lstm": {
            "n_epochs": [3, 5],
            "hidden_dim": [64, 128],
            "lr": [0.001, 0.005],
        },
        "embedding": {
            "C": [0.1, 1.0, 10.0],
        }
    }
    return defaults.get(name, {})


