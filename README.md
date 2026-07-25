# 🛡️ ImpreX Sentinel: Explainable Multi-Channel Communication Threat Intelligence Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ImpreX Sentinel** is an advanced, NLP-driven threat classification engine engineered to detect and analyze digital communication threats across **SMS, WhatsApp, and Email** channels. Operating as a core security module within the ImpreX ecosystem, the engine evaluates incoming messages and categorizes them into three distinct risk tiers: **Legitimate (Ham)**, **Suspicious (Spam)**, and **Critical Threat (Phishing/Smishing)**.

To overcome the "black box" limitation of deep neural networks, ImpreX Sentinel integrates **Explainable AI (XAI)** powered by **LIME (Local Interpretable Model-agnostic Explanations)**. This empowers security analysts and end-users with real-time, token-level attribution showing the exact linguistic clues and keywords driving each threat decision.

---

## 🌟 Key Features

- **3-Tier Risk Taxonomy**: Categorizes communications into `0: Ham (Legitimate)`, `1: Spam (Promotional)`, and `2: Phishing/Smishing (Critical Threat)`.
- **URL Isolation Strategy**: Automatically replaces hyperlinks with a generic `urltoken` placeholder, forcing the neural network to analyze linguistic and context patterns surrounding link placements rather than relying solely on external domain blacklists.
- **Deep Learning GRU Engine**: Powered by a Gated Recurrent Unit (GRU) neural network with post-padded sequences (`maxlen=150`) and weighted Cross-Entropy Loss to handle class imbalance across SMS and Email lengths.
- **Baseline Naive Bayes Classifier**: High-speed fallback model trained on TF-IDF features (92.98% accuracy).
- **Explainable AI (LIME Integration)**: Generates feature contribution charts and inline token highlights (flagging keywords like *urgent*, *verify*, *account*, *suspended*, *credentials*).
- **Streamlit Deployment Ready**: Built-in 60-30-10 palette UI/UX dashboard supporting real-time single message scanning and bulk CSV processing.

---

## 📊 Benchmark Metrics (571,780 Unified Messages Ingested)

| Model Architecture | Training Samples | Overall Accuracy | Phishing Precision | Key Performance Highlights |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Naive Bayes (TF-IDF)** | 457,424 | **92.98%** | **0.77** | High precision on Spam (0.99) & Ham (0.96) |
| **Lightweight GRU Network (DL)** | 50,000 | **94.33%** | **0.80** | Peak validation accuracy of **94.75%** with Early Stopping |

### Dataset Class Distribution
- **Ham (Class 0)**: 212,605 samples (Class Weight: `0.896`)
- **Spam (Class 1)**: 254,191 samples (Class Weight: `0.750`)
- **Phishing (Class 2)**: 104,984 samples (Class Weight: `1.815`)

---

## 🏗️ Repository & Directory Structure

```text
├── .streamlit/
│   └── config.toml             # Streamlit 60-30-10 color theme (60% White, 30% Crimson, 10% Green)
├── app_mark3.py                 # Primary Streamlit Dashboard Application
├── app.py                       # Streamlit entrypoint wrapper
├── preprocessing.py             # Regex URL masking ('urltoken'), normalization, & label mapping
├── models.py                    # GRU PyTorch Neural Network & Naive Bayes Baseline models
├── explainability.py            # LIME Explainable AI pipeline wrapper
├── train_mark3.py               # Dataset ingestion & model training script
├── spam_detection_mark3.ipynb   # Interactive demonstration & evaluation notebook
├── mark3_gru_model.pth          # Pre-trained GRU weights
├── tokenizer.pickle             # Tokenizer vocabulary dictionary (maxlen=150)
├── tfidf_vectorizer.pickle      # TF-IDF matrix vectorizer
├── naive_bayes_model.pkl        # Baseline Naive Bayes model
├── requirements.txt             # Python dependencies for Streamlit Cloud deployment
└── README.md                    # Module documentation
```

---

## ⚙️ Quickstart & Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Sanil-Samir-Mhatre/ImpreX-Sentinel-Explainable-Multi-Channel-Communication-Threat-Intelligence-Engine.git
cd ImpreX-Sentinel-Explainable-Multi-Channel-Communication-Threat-Intelligence-Engine
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
```bash
python -m streamlit run app_mark3.py
```

---

## 🚀 Deploying to Streamlit Community Cloud

1. Fork or push this repository to GitHub.
2. Log in to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click **New app**, select your repository: `Sanil-Samir-Mhatre/ImpreX-Sentinel-Explainable-Multi-Channel-Communication-Threat-Intelligence-Engine`.
4. Set **Main file path** to `app_mark3.py` (or `app.py`).
5. Click **Deploy!**

---

## 🎨 UI/UX Design System (60-30-10 Palette)

- **60% Base Background**: Pure White (`#FFFFFF`) & Off-White (`#F8F9FA`) for clean metric cards.
- **30% Primary Interactive Elements**: Crimson Red (`#D9381E`) for buttons, active tabs, and `.badge-critical` threat alerts.
- **10% Indicator Accent**: Emerald Green (`#10B981`) for `.badge-safe` legitimate communication badges.

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
