"""
Text preprocessing pipeline using TF-IDF vectorization + label encoding.
Scikit-learn compatible — returns sparse matrices instead of padded sequences.
"""

import re
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from typing import Tuple

# Custom Hinglish Stopwords to remove grammatical noise
HINGLISH_STOPWORDS = {
    "hai", "he", "ho", "tha", "thi", "the", "ko", "se", "aur", "ki", "ka", "ke", 
    "par", "pe", "me", "main", "bhi", "hi", "toh", "to", "jo", "kar", "karo", 
    "karna", "krna", "gaya", "gayi", "gaye", "hua", "hui", "hue", "ab", "kab", 
    "tab", "jab", "aur", "ya", "par", "lekin", "hi", "kuch", "sab", "apna", 
    "apne", "meri", "mera", "mere", "tum", "aap", "hum", "woh", "yeh", "hi", 
    "hoga", "hogi", "hoge", "karke", "krke", "sath", "saath", "liye", "bina", 
    "pehle", "baad", "andar", "bahar"
}

class TextPreprocessor:
    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.vectorizer = None
        self.label_encoder = LabelEncoder()

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize and clean raw text, filtering out custom Hinglish stopwords."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", " URL ", text)
        text = re.sub(r"\$\d+\.?\d*", " AMOUNT ", text)
        text = re.sub(r"#\d+", " ORDERID ", text)
        text = re.sub(r"\S+@\S+\.\S+", " EMAIL ", text)
        text = re.sub(r"\d+", " NUM ", text)
        text = re.sub(r"[^A-Za-z0-9\s\.\,\!\?]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        # Filter Hinglish stopwords
        words = [w for w in text.split() if w not in HINGLISH_STOPWORDS]
        return " ".join(words)

    def fit(self, texts: pd.Series, labels: pd.Series) -> "TextPreprocessor":
        cleaned = texts.astype(str).apply(self.clean_text).tolist()
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            strip_accents="unicode",
            lowercase=True,
        )
        self.vectorizer.fit(cleaned)
        self.label_encoder.fit(labels)
        return self

    def transform(self, texts: pd.Series) -> csr_matrix:
        cleaned = texts.astype(str).apply(self.clean_text).tolist()
        return self.vectorizer.transform(cleaned)

    def fit_transform(self, texts: pd.Series, labels: pd.Series) -> Tuple[csr_matrix, np.ndarray]:
        self.fit(texts, labels)
        X = self.transform(texts)
        y = self.encode_labels(labels)
        return X, y

    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        return self.label_encoder.transform(labels)

    def decode_labels(self, encoded) -> list:
        return self.label_encoder.inverse_transform(encoded)

    def get_feature_count(self) -> int:
        return len(self.vectorizer.vocabulary_) if self.vectorizer else 0

    def get_classes(self) -> list:
        return list(self.label_encoder.classes_)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "label_encoder": self.label_encoder,
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "min_df": self.min_df,
                "max_df": self.max_df,
                "sublinear_tf": self.sublinear_tf,
            }, f)

    @classmethod
    def load(cls, path: str) -> "TextPreprocessor":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            max_features=data["max_features"],
            ngram_range=data["ngram_range"],
            min_df=data["min_df"],
            max_df=data["max_df"],
            sublinear_tf=data["sublinear_tf"],
        )
        obj.vectorizer = data["vectorizer"]
        obj.label_encoder = data["label_encoder"]
        return obj

def load_data(train_path: str, val_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(train_path),
        pd.read_csv(val_path),
        pd.read_csv(test_path),
    )
