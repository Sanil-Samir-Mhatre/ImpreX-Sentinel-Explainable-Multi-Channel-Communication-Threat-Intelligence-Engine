"""
Production Training Script for Mark3 (3-Class Spam, Ham, Phishing Threat Classification)

Fulfills all pipeline requirements:
1. Data Ingestion via KaggleHub & Regex URL Masking ('urltoken' preservation)
2. Stratified Split & Class Weight balancing for SMS/Email discrepancies
3. Baseline ML Model (TF-IDF + Naive Bayes) with Confusion Matrix & Classification Report
4. Lightweight DL Model (GRU Neural Network, maxlen=150, Early Stopping)
5. Explainable AI (LIME) token importance visualization
6. Model Artifact Persistence (.pkl / .pth)
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import kagglehub

from preprocessing import standardize_labels, calculate_class_weights, clean_text
from models import BaselineMLModel, GRUClassifier
from explainability import ThreatExplainer

def find_csv_files(dataset_path):
    """Finds all CSV files within a downloaded KaggleHub directory."""
    csv_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def load_and_merge_datasets():
    print("=" * 60, flush=True)
    print("1. DATA PREPARATION & FEATURE ENGINEERING", flush=True)
    print("=" * 60, flush=True)

    # Download Dataset 1: akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000
    print("\nDownloading Dataset 1 (akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000)...", flush=True)
    path1 = kagglehub.dataset_download("akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000")
    csvs1 = find_csv_files(path1)

    # Download Dataset 2: dharshiyanacc/spam-ham-and-phishing-message-dataset-for-nlp
    print("\nDownloading Dataset 2 (dharshiyanacc/spam-ham-and-phishing-message-dataset-for-nlp)...", flush=True)
    path2 = kagglehub.dataset_download("dharshiyanacc/spam-ham-and-phishing-message-dataset-for-nlp")
    csvs2 = find_csv_files(path2)

    dfs = []

    # Ingest CSVs from Dataset 1
    for csv_file in csvs1:
        print(f"Loading {os.path.basename(csv_file)}...", flush=True)
        df = pd.read_csv(csv_file)
        text_cols = [c for c in df.columns if 'text' in c.lower() or 'message' in c.lower() or 'email' in c.lower() or 'body' in c.lower()]
        label_cols = [c for c in df.columns if 'label' in c.lower() or 'target' in c.lower() or 'category' in c.lower() or 'type' in c.lower() or 'class' in c.lower()]
        
        if text_cols and label_cols:
            std_df = standardize_labels(df, text_cols[0], label_cols[0])
            dfs.append(std_df)

    # Ingest CSVs from Dataset 2
    for csv_file in csvs2:
        print(f"Loading {os.path.basename(csv_file)}...", flush=True)
        df = pd.read_csv(csv_file)
        text_cols = [c for c in df.columns if 'text' in c.lower() or 'message' in c.lower() or 'email' in c.lower() or 'body' in c.lower()]
        label_cols = [c for c in df.columns if 'label' in c.lower() or 'target' in c.lower() or 'category' in c.lower() or 'type' in c.lower() or 'class' in c.lower()]
        
        if text_cols and label_cols:
            std_df = standardize_labels(df, text_cols[0], label_cols[0])
            dfs.append(std_df)

    if not dfs:
        raise ValueError("Could not find suitable text and label columns in downloaded datasets.")

    combined_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    print(f"\nTotal Merged Samples: {len(combined_df)}", flush=True)
    print("\nClass Distribution:", flush=True)
    label_names = {0: 'Ham', 1: 'Spam', 2: 'Phishing'}
    for k, v in combined_df['target'].value_counts().items():
        print(f"  {label_names.get(k, k)} (Class {k}): {v} samples", flush=True)

    return combined_df

def main():
    df = load_and_merge_datasets()

    X = df['cleaned_text'].values
    y = df['target'].values

    # Stratified Train-Test Split (80% Train, 10% Val, 10% Test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    # Compute Class Weights to tackle class imbalance
    class_weights = calculate_class_weights(y_train)
    print("\nCalculated Class Weights:", class_weights, flush=True)

    # -------------------------------------------------------------
    # 2. BASELINE MACHINE LEARNING MODEL
    # -------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("2. BASELINE MACHINE LEARNING MODEL (TF-IDF + Naive Bayes)", flush=True)
    print("=" * 60, flush=True)
    
    baseline = BaselineMLModel(max_features=10000)
    print("Training Naive Bayes Baseline Classifier...", flush=True)
    baseline.train(X_train, y_train)

    acc, report, cm = baseline.evaluate(X_test, y_test)
    print(f"\nBaseline Model Accuracy: {acc * 100:.2f}%", flush=True)
    print("\nClassification Report:\n", report, flush=True)

    # Save Baseline Artifacts
    baseline.save('tfidf_vectorizer.pickle', 'naive_bayes_model.pkl')
    print("Saved 'tfidf_vectorizer.pickle' and 'naive_bayes_model.pkl'", flush=True)

    # -------------------------------------------------------------
    # 3. LIGHTWEIGHT DEEP LEARNING MODEL (GRU)
    # -------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("3. LIGHTWEIGHT DEEP LEARNING MODEL (Embedding + GRU)", flush=True)
    print("=" * 60, flush=True)

    # Limit training sample if dataset is massive for quick demonstration
    max_train_samples = min(30000, len(X_train))
    X_tr_dl = X_train[:max_train_samples]
    y_tr_dl = y_train[:max_train_samples]

    gru_classifier = GRUClassifier(vocab_size=10000, embed_dim=128, hidden_dim=64, num_classes=3, max_len=150)
    print(f"Training GRU Network on {len(X_tr_dl)} samples with maxlen=150 (post padding/truncating)...", flush=True)
    
    gru_classifier.train(
        X_tr_dl, y_tr_dl,
        X_val, y_val,
        class_weights=class_weights,
        epochs=5,
        batch_size=64,
        patience=2
    )

    # Evaluate Deep Learning Model on Test Set
    dl_preds = gru_classifier.predict(X_test)
    dl_acc = accuracy_score(y_test, dl_preds)
    dl_report = classification_report(y_test, dl_preds, target_names=['Ham', 'Spam', 'Phishing'])
    print(f"\nDeep Learning GRU Accuracy: {dl_acc * 100:.2f}%", flush=True)
    print("\nDeep Learning Classification Report:\n", dl_report, flush=True)

    # Save Deep Learning Artifacts
    gru_classifier.save('mark3_gru_model.pth', 'tokenizer.pickle')
    print("Saved 'mark3_gru_model.pth' and 'tokenizer.pickle'", flush=True)

    # -------------------------------------------------------------
    # 4. EXPLAINABLE AI (XAI - LIME)
    # -------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("4. EXPLAINABLE AI (XAI) VISUALIZATION WITH LIME", flush=True)
    print("=" * 60, flush=True)

    sample_phishing_msg = "URGENT SECURITY NOTICE: Your bank account has been suspended due to suspicious activity. Verify credentials immediately at http://secure-verify-bank.com/login"
    
    explainer = ThreatExplainer(gru_classifier.predict_proba)
    exp, target_class = explainer.explain_instance(sample_phishing_msg, num_features=8)

    print(f"\nSample Threat Input:\n  '{sample_phishing_msg}'", flush=True)
    print(f"Predicted Class: {['Ham', 'Spam', 'Phishing'][target_class]}", flush=True)
    print("\nTop Contributing Tokens:", flush=True)
    for token, score in exp.as_list():
        print(f"  Token: '{token:<15}' Importance Weight: {score:+.4f}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("MARK3 TRAINING PIPELINE COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 60, flush=True)

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
