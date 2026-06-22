try:
    from . import dll_loader
except ImportError:
    import dll_loader

import re
from collections import Counter
import torch
import torch.nn as nn


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class Vocabulary:
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        
    def fit(self, texts):
        word_counts = Counter()
        for text in texts:
            word_counts.update(re.findall(r'\w+', str(text).lower()))
        
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        for i, (word, _) in enumerate(most_common):
            idx = i + 2
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
    def transform(self, texts, max_len=50):
        sequences = []
        for text in texts:
            seq = []
            for word in re.findall(r'\w+', str(text).lower()):
                seq.append(self.word2idx.get(word, 1)) # 1 is UNK
            # Pad or truncate (pre-padding)
            if len(seq) < max_len:
                seq = [0] * (max_len - len(seq)) + seq
            else:
                seq = seq[:max_len]
            sequences.append(seq)
        return sequences

class LSTMNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x) # [batch_size, seq_len, embedding_dim]
        # lstm returns output and (hidden, cell)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Max pooling over the sequence dimension (dim=1)
        pooled, _ = torch.max(lstm_out, dim=1)
        out = self.fc(pooled)
        return out

class LSTMClassifier(BaseEstimator, ClassifierMixin):
    # Specify that this is a sequence model so that the pipeline passes raw text rather than TF-IDF
    is_sequence_model = True

    def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=128, n_epochs=5, batch_size=64, lr=0.001, max_len=50, num_layers=1, random_state=42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_len = max_len
        self.num_layers = num_layers
        self.random_state = random_state
        self.vocab = None
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        # Set seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Handle conversion if X is pandas Series
        X_list = list(X)
        
        # Determine classes (need labels to start from 0 to num_classes-1)
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        
        # Build vocabulary
        self.vocab = Vocabulary(self.vocab_size)
        self.vocab.fit(X_list)
        
        # Convert texts to sequences
        seqs = self.vocab.transform(X_list, max_len=self.max_len)
        X_tensor = torch.tensor(seqs, dtype=torch.long)
        y_tensor = torch.tensor(list(y), dtype=torch.long)
        
        # Initialize network
        self.model = LSTMNet(
            vocab_size=len(self.vocab.word2idx),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=num_classes,
            num_layers=self.num_layers
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Simple training loop
        self.model.train()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.n_epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_list = list(X)
        seqs = self.vocab.transform(X_list, max_len=self.max_len)
        X_tensor = torch.tensor(seqs, dtype=torch.long)
        
        probs = []
        with torch.no_grad():
            dataset = torch.utils.data.TensorDataset(X_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            for batch_x, in dataloader:
                logits = self.model(batch_x)
                prob = torch.softmax(logits, dim=1).numpy()
                probs.append(prob)
                
        return np.vstack(probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        return preds
