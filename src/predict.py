import sys
import os

# Add 'src' directory to sys.path to ensure pickled models can resolve module names
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    import src.dll_loader
except ImportError:
    try:
        import dll_loader
    except ImportError:
        pass

import numpy as np
import joblib
from typing import List, Dict

try:
    from .data_preprocessing import TextPreprocessor
except ImportError:  # Support direct execution: python src/predict.py
    from data_preprocessing import TextPreprocessor


class TicketClassifier:
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model = joblib.load(model_path)
        self.pre = TextPreprocessor.load(preprocessor_path)
        self.classes_ = self.pre.get_classes()

    def predict(self, texts: List[str], top_k: int = 3) -> List[Dict]:
        if hasattr(self.model, "is_sequence_model") and self.model.is_sequence_model:
            cleaned = [self.pre.clean_text(t) for t in texts]
            preds = self.model.predict(cleaned)
            probs = self.model.predict_proba(cleaned)
        else:
            X = self.pre.transform(__import__("pandas").Series(texts))
            preds = self.model.predict(X)
            # Probabilities if available
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)
            else:
                # Fall back to decision_function or one-hot
                if hasattr(self.model, "decision_function"):
                    scores = self.model.decision_function(X)
                    # softmax for normalization
                    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
                    probs = exp / exp.sum(axis=1, keepdims=True)
                else:
                    probs = np.eye(len(self.classes_))[preds]


        results = []
        for text, pred, prob in zip(texts, preds, probs):
            top_idx = np.argsort(prob)[::-1][:top_k]
            results.append({
                "text": text,
                "prediction": self.classes_[pred],
                "confidence": float(prob[pred]),
                "top_k": [
                    {"label": self.classes_[i], "score": float(prob[i])}
                    for i in top_idx
                ],
            })
        return results


if __name__ == "__main__":
    clf = TicketClassifier("artifacts/best_model.joblib", "artifacts/preprocessor.pkl")
    samples = [
        "My card was charged twice for order #45213, please help!",
        "The app keeps crashing whenever I open the dashboard.",
        "Can I get a refund for my damaged package?",
        "I want to know if the pro plan supports SSO login.",
        "Where is my order? It's been 8 days and still says in transit.",
    ]
    for r in clf.predict(samples, top_k=3):
        print(f"\nText: {r['text']}")
        print(f"   -> {r['prediction']} ({r['confidence']:.2%})")
        for alt in r["top_k"]:
            print(f"     * {alt['label']}: {alt['score']:.2%}")
