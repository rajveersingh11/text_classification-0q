import os
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

# Ensure DLL loader runs first
try:
    import src.dll_loader
except ImportError:
    try:
        import dll_loader
    except ImportError:
        pass

from sentence_transformers import SentenceTransformer

class EmbeddingClassifier(BaseEstimator, ClassifierMixin):
    is_sequence_model = True  # We want raw texts passed

    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", C=1.0, random_state=42):
        self.model_name = model_name
        self.C = C
        self.random_state = random_state
        self.encoder = None
        self.classifier = None
        self.classes_ = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if "encoder" in state:
            state["encoder"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.encoder = None

    def fit(self, X, y):
        # Load sentence transformer
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.model_name)
            
        # Encode text queries into dense vectors
        X_list = list(X)
        print(f"Encoding {len(X_list)} samples with {self.model_name}...")
        X_emb = self.encoder.encode(X_list, show_progress_bar=False, convert_to_numpy=True)
        
        # Fit logistic regression on embeddings
        self.classes_ = np.unique(y)
        self.classifier = LogisticRegression(C=self.C, random_state=self.random_state, max_iter=1000, n_jobs=1)
        self.classifier.fit(X_emb, y)
        return self

    def predict_proba(self, X):
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.model_name)
        X_list = list(X)
        X_emb = self.encoder.encode(X_list, show_progress_bar=False, convert_to_numpy=True)
        return self.classifier.predict_proba(X_emb)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
