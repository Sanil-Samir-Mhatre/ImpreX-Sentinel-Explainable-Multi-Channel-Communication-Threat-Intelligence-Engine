"""
Models Module for Mark3: Baseline ML (TF-IDF + Naive Bayes) & Lightweight Deep Learning (GRU)

Handles:
- Classical ML model initialization, training, evaluation, and confusion matrix rendering.
- Deep Learning GRU neural network (PyTorch / Keras) with Embedding, GRU, Dropout, Softmax output.
- Text Tokenizer utility supporting padding/truncating at maxlen=150.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from preprocessing import clean_text

# Text Tokenizer Utility for Deep Learning
class SimpleTextTokenizer:
    """Lightweight custom Tokenizer matching Keras interface for PyTorch/DL pipeline."""
    def __init__(self, num_words=10000, oov_token="<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {oov_token: 1}
        self.index_word = {1: oov_token}
        self.word_counts = {}

    def fit_on_texts(self, texts):
        counts = {}
        for text in texts:
            for word in text.split():
                counts[word] = counts.get(word, 0) + 1
        
        # Sort words by frequency
        sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, count) in enumerate(sorted_words[:self.num_words - 2], start=2):
            self.word_index[word] = idx
            self.index_word[idx] = word
            self.word_counts[word] = count

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = []
            for word in text.split():
                seq.append(self.word_index.get(word, 1)) # 1 is OOV token
            sequences.append(seq)
        return sequences

    def pad_sequences(self, sequences, maxlen=150, padding='post', truncating='post'):
        padded = np.zeros((len(sequences), maxlen), dtype=int)
        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                continue
            if truncating == 'post':
                seq = seq[:maxlen]
            else:
                seq = seq[-maxlen:]
            
            if padding == 'post':
                padded[i, :len(seq)] = seq
            else:
                padded[i, -len(seq):] = seq
        return padded

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# Baseline Machine Learning Model
class BaselineMLModel:
    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.classifier = MultinomialNB(alpha=0.1)

    def train(self, X_train, y_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vec, y_train)
        return X_train_vec

    def predict(self, X_test):
        X_test_vec = self.vectorizer.transform(X_test)
        return self.classifier.predict(X_test_vec)

    def predict_proba(self, X_test):
        X_test_vec = self.vectorizer.transform(X_test)
        return self.classifier.predict_proba(X_test_vec)

    def evaluate(self, X_test, y_test, target_names=['Ham', 'Spam', 'Phishing']):
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, target_names=target_names)
        cm = confusion_matrix(y_test, preds)
        return acc, report, cm

    def save(self, vec_path, model_path):
        with open(vec_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(model_path, 'wb') as f:
            pickle.dump(self.classifier, f)

# Deep Learning GRU PyTorch Neural Network
class GRUNetwork(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=64, num_classes=3, dropout=0.2):
        super(GRUNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 2, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        out, h_n = self.gru(embedded)
        # Take the last hidden state output
        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits

class GRUClassifier:
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=64, num_classes=3, max_len=150):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GRUNetwork(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(self.device)
        self.tokenizer = SimpleTextTokenizer(num_words=vocab_size)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def train(self, X_train, y_train, X_val, y_val, class_weights=None, epochs=10, batch_size=64, patience=3):
        # Fit tokenizer on training texts
        self.tokenizer.fit_on_texts(X_train)
        
        train_seqs = self.tokenizer.texts_to_sequences(X_train)
        val_seqs = self.tokenizer.texts_to_sequences(X_val)

        X_tr_pad = self.tokenizer.pad_sequences(train_seqs, maxlen=self.max_len, padding='post', truncating='post')
        X_va_pad = self.tokenizer.pad_sequences(val_seqs, maxlen=self.max_len, padding='post', truncating='post')

        # DataLoader setup
        train_ds = TensorDataset(torch.tensor(X_tr_pad, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_va_pad, dtype=torch.long), torch.tensor(y_val, dtype=torch.long))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Loss function with Class Weights
        if class_weights is not None:
            weights_tensor = torch.tensor([class_weights[c] for c in range(3)], dtype=torch.float32).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y_b)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_b).sum().item()
                total += len(y_b)

            train_loss = total_loss / total
            train_acc = correct / total

            # Validation step
            self.model.eval()
            val_loss_total, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for x_b, y_b in val_loader:
                    x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                    logits = self.model(x_b)
                    loss = criterion(logits, y_b)
                    val_loss_total += loss.item() * len(y_b)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == y_b).sum().item()
                    val_total += len(y_b)

            val_loss = val_loss_total / val_total
            val_acc = val_correct / val_total

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'mark3_gru_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load best model weights and clean up temporary file
        self.model.load_state_dict(torch.load('mark3_gru_best.pth'))
        if os.path.exists('mark3_gru_best.pth'):
            os.remove('mark3_gru_best.pth')

    def predict_proba(self, texts):
        """Returns softmax probabilities (N, 3) for raw or preprocessed input texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        # 1. Apply full preprocessing pipeline (URL masking to 'urltoken', lowercasing, stopword removal)
        cleaned_texts = [clean_text(t) if isinstance(t, str) else "" for t in texts]

        # 2. Tokenize and pad sequences to maxlen=150
        self.model.eval()
        seqs = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded = self.tokenizer.pad_sequences(seqs, maxlen=self.max_len, padding='post', truncating='post')
        
        x_tensor = torch.tensor(padded, dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, texts):
        """Returns discrete integer class labels (0: Ham, 1: Spam, 2: Phishing) of shape (N,)."""
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1)

    def save(self, model_path, tokenizer_path):
        torch.save(self.model.state_dict(), model_path)
        self.tokenizer.save(tokenizer_path)
