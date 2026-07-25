"""
Preprocessing & Feature Engineering Module for Mark3 (3-Class Spam/Ham/Phishing)

Key Design Considerations (per Literature & Best Practices):
1. URL Masking: Replaces URLs with 'urltoken' so alphanumeric regex cleaning
   does not strip '<' and '>' brackets.
2. Normalization: Lowercases text, cleans punctuation (preserving 'urltoken'),
   and removes English stopwords.
3. Label Standardization: Harmonizes labels to 0 (Ham), 1 (Spam), 2 (Phishing).
"""

import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

# Download NLTK stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))

# Regex pattern for robust URL matching
URL_REGEX = re.compile(
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
    r'www\.[a-zA-Z0-9.\-]+(?:\.[a-zA-Z]{2,})+|'
    r'[a-zA-Z0-9.\-]+\.(?:com|org|net|edu|gov|mil|io|ai|co|info|biz|me|xyz|tech|online)(?:/[^\s]*)?',
    re.IGNORECASE
)

def mask_urls(text: str, token: str = "urltoken") -> str:
    """Replaces URLs/links in text with a generic token."""
    if not isinstance(text, str):
        return ""
    return URL_REGEX.sub(f" {token} ", text)

def clean_text(text: str, remove_stopwords: bool = True, token: str = "urltoken") -> str:
    """
    Normalizes text:
    - Replaces URLs with 'urltoken'
    - Lowercases text
    - Retains letters, numbers, whitespace, and 'urltoken'
    - Removes English stopwords
    """
    if not isinstance(text, str):
        return ""

    # 1. URL Masking
    text = mask_urls(text, token=token)

    # 2. Lowercasing
    text = text.lower()

    # 3. Clean special chars (preserving alphanumeric and whitespace)
    # Note: 'urltoken' is purely alphabetic so it will NOT be stripped!
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # 4. Stopwords filtering & whitespace collapse
    words = text.split()
    if remove_stopwords:
        words = [w for w in words if w not in STOP_WORDS or w == token]

    return " ".join(words)

def standardize_labels(df: pd.DataFrame, text_col: str, label_col: str, max_samples: int = 50000) -> pd.DataFrame:
    """
    Standardizes label mapping across disparate datasets into:
    0: Ham
    1: Spam
    2: Phishing
    """
    df = df.copy()
    df = df.dropna(subset=[text_col, label_col])

    # If dataset is massive, sample for fast interactive execution
    if len(df) > max_samples:
        print(f"Sampling {max_samples} records from {len(df)} total rows for high-speed processing...", flush=True)
        df = df.sample(n=max_samples, random_state=42)

    # Convert label to string for uniform mapping
    labels_str = df[label_col].astype(str).str.lower().str.strip()

    mapping = {
        'ham': 0, 'legitimate': 0, '0': 0, '0.0': 0, 'normal': 0, 'benign': 0,
        'spam': 1, '1': 1, '1.0': 1, 'junk': 1,
        'phishing': 2, 'phish': 2, 'smishing': 2, '2': 2, '2.0': 2, 'malicious': 2
    }

    df['target'] = labels_str.map(mapping)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)
    
    print(f"Cleaning {len(df)} text records with regex URL masking ('urltoken')...", flush=True)
    df['cleaned_text'] = df[text_col].apply(clean_text)

    # Filter out empty cleaned text
    df = df[df['cleaned_text'].str.strip().str.len() > 0]
    return df[['cleaned_text', 'target']]

def calculate_class_weights(y_train: np.ndarray) -> dict:
    """Computes balanced class weights for training under class imbalance."""
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))
