"""
Streamlit Threat Intelligence Dashboard - IMPREX Mark3
3-Class Communications Threat Classifier (0: Ham, 1: Spam, 2: Phishing)

Design System:
- 60% Base: Pure White (#FFFFFF)
- 30% Primary Interactive Accent: Crimson Red (#D9381E)
- 10% Indicator Accent: Emerald Green (#10B981)
"""

import os
import re
import html
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from preprocessing import clean_text, mask_urls
from models import BaselineMLModel, GRUClassifier
from explainability import ThreatExplainer

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME SETUP
# -----------------------------------------------------------------------------
model_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(model_dir, "Logo_Retroify.png")

st.set_page_config(
    page_title="IMPREX Sentinel | Threat Intelligence",
    page_icon=logo_path if os.path.exists(logo_path) else "🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for 60-30-10 Palette & Premium Aesthetics
CUSTOM_CSS = """
<style>
    /* Metric Cards Styling */
    .metric-card {
        background-color: #F8F9FA;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 12px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-card .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-card .metric-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748B;
        margin-bottom: 4px;
    }

    /* Threat Status Badges */
    .badge-critical {
        background-color: #D9381E !important;
        color: #FFFFFF !important;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 700;
        font-size: 15px;
        display: inline-block;
        box-shadow: 0 0 12px rgba(217, 56, 30, 0.4);
        letter-spacing: 0.5px;
    }
    .badge-spam {
        background-color: #F59E0B !important;
        color: #FFFFFF !important;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 700;
        font-size: 15px;
        display: inline-block;
        box-shadow: 0 0 10px rgba(245, 158, 11, 0.3);
        letter-spacing: 0.5px;
    }
    .badge-safe {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 700;
        font-size: 15px;
        display: inline-block;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.3);
        letter-spacing: 0.5px;
    }

    /* Token Risk Highlighting */
    .token-highlight-critical {
        background-color: rgba(217, 56, 30, 0.18);
        color: #991B1B;
        border-bottom: 2px solid #D9381E;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 700;
    }
    .token-highlight-url {
        background-color: rgba(16, 185, 129, 0.18);
        color: #065F46;
        border-bottom: 2px solid #10B981;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 700;
    }

    /* Tab & Expander Modernization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 2px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F8F9FA;
        border-radius: 8px 8px 0px 0px;
        color: #64748B;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #D9381E !important;
        border-bottom: 3px solid #D9381E !important;
    }
    .streamlit-expanderHeader {
        background-color: #F8F9FA;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL & ARTIFACT CACHED LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Loads GRU Deep Learning network, Tokenizer, Naive Bayes, and TF-IDF matrix."""
    model_dir = os.path.dirname(os.path.abspath(__file__))
    
    gru_weights_path = os.path.join(model_dir, 'mark3_gru_model.pth')
    tokenizer_path = os.path.join(model_dir, 'tokenizer.pickle')
    nb_path = os.path.join(model_dir, 'naive_bayes_model.pkl')
    vec_path = os.path.join(model_dir, 'tfidf_vectorizer.pickle')

    gru_classifier = None
    baseline_model = None

    # Load GRU Classifier
    if os.path.exists(gru_weights_path) and os.path.exists(tokenizer_path):
        try:
            gru_classifier = GRUClassifier(vocab_size=10000, embed_dim=128, hidden_dim=64, num_classes=3, max_len=150)
            gru_classifier.model.load_state_dict(torch.load(gru_weights_path, map_location=gru_classifier.device))
            gru_classifier.tokenizer = gru_classifier.tokenizer.load(tokenizer_path)
        except Exception as e:
            st.error(f"Error loading GRU model: {e}")

    # Load Baseline Model
    if os.path.exists(nb_path) and os.path.exists(vec_path):
        try:
            baseline_model = BaselineMLModel()
            with open(vec_path, 'rb') as f:
                baseline_model.vectorizer = pickle.load(f)
            with open(nb_path, 'rb') as f:
                baseline_model.classifier = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading Baseline model: {e}")

    return gru_classifier, baseline_model

gru_classifier, baseline_model = load_models()

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & PRESET SELECTION
# -----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists(logo_path):
        st.image(logo_path, use_column_width=True)
    else:
        st.image("https://img.icons8.com/color/96/shield-with-signature.png", width=70)
    st.title(" ")
    st.caption(" ")
    
    st.divider()
    
    engine_choice = st.selectbox(
        "🧠 Detection Model Engine",
        ["Deep Learning GRU (Recommended)", "Baseline Naive Bayes (TF-IDF)"]
    )
    
    confidence_threshold = st.slider("⚠️ Phishing Warning Sensitivity", 0.30, 0.90, 0.50, 0.05)
    
    st.divider()
    st.subheader("📋 Threat Test Presets")
    preset = st.radio(
        "Load sample message:",
        ["None", "Phishing Bank Scam", "SMS Spam Promotion", "Legitimate Work Email"]
    )
    
    st.divider()
    st.markdown("""
    **Architecture Specifications:**
    - **Taxonomy**: 3-Class (Ham / Spam / Phishing)
    - **URL Isolation**: `<URL>` tokenized via `urltoken`
    - **Sequence Maxlen**: 150 tokens (Post-padded)
    - **Loss Function**: Weighted Cross-Entropy
    """)

# Preset Message Text Loader
PRESETS = {
    "Phishing Bank Scam": "URGENT SECURITY NOTICE: Your bank account has been suspended due to suspicious activity. Verify credentials immediately at http://secure-verify-bank.com/login",
    "SMS Spam Promotion": "Congratulations! You have won a $1,000 Walmart gift card. Claim your reward now by clicking http://free-gift-promo.xyz/claim",
    "Legitimate Work Email": "Hi Alex, please find attached the quarterly financial summary report for your review. Let me know if you have any questions before the afternoon meeting."
}

default_text = PRESETS.get(preset, "") if preset != "None" else ""

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD UI
# -----------------------------------------------------------------------------
st.title("🛡️ IMPREX Sentinel — 3-Class NLP Threat Intelligence Engine")
st.markdown("Automated **3-Class Semantic Text Analysis** isolating link reputation checks to focus strictly on linguistic threat context.")

# Top Metrics Row
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.markdown('<div class="metric-card"><div class="metric-label">Model Status</div><div class="metric-val" style="color: #10B981;">ACTIVE</div></div>', unsafe_allow_html=True)
with mcol2:
    st.markdown('<div class="metric-card"><div class="metric-label">Classification Taxonomy</div><div class="metric-val">3-Class</div></div>', unsafe_allow_html=True)
with mcol3:
    st.markdown('<div class="metric-card"><div class="metric-label">Target Maxlen</div><div class="metric-val">150 Tokens</div></div>', unsafe_allow_html=True)
with mcol4:
    st.markdown('<div class="metric-card"><div class="metric-label">URL Strategy</div><div class="metric-val" style="color: #D9381E;">Tokenized</div></div>', unsafe_allow_html=True)

st.divider()

# Main Application Tabs
tab_scan, tab_batch, tab_analytics = st.tabs(["🔍 Real-Time Scan", "📁 Batch Processing", "📈 Architecture Benchmarks"])

# -----------------------------------------------------------------------------
# TAB 1: REAL-TIME THREAT SCAN
# -----------------------------------------------------------------------------
with tab_scan:
    col_input, col_output = st.columns([1.1, 0.9], gap="large")
    
    with col_input:
        st.subheader("1. Message Content Input")
        user_input = st.text_area(
            "Paste SMS text or email message body below:",
            value=default_text,
            height=200,
            placeholder="Type or paste suspicious email body or SMS message..."
        )
        
        scan_btn = st.button("🚨 SCAN MESSAGE FOR THREATS", type="primary", use_container_width=True)
        
    if scan_btn or user_input.strip():
        if not user_input.strip():
            with col_output:
                st.warning("Please enter text or select a sample preset.")
        else:
            with st.spinner("Analyzing linguistic patterns & token weights..."):
                # Preprocess input text
                cleaned = clean_text(user_input)
                masked = mask_urls(user_input, token="urltoken")
                
                # Predict using selected engine
                if "GRU" in engine_choice and gru_classifier is not None:
                    probs = gru_classifier.predict_proba(user_input)[0]
                elif baseline_model is not None:
                    probs = baseline_model.predict_proba([cleaned])[0]
                else:
                    st.error("Model engine artifacts not found. Please run train_mark3.py.")
                    st.stop()
                
                pred_class = int(probs.argmax())
                ham_prob, spam_prob, phish_prob = probs[0], probs[1], probs[2]
                
                # Determine Status Badge
                if pred_class == 2 or phish_prob >= confidence_threshold:
                    status_html = '<div class="badge-critical">CRITICAL THREAT: PHISHING / SMISHING DETECTED</div>'
                elif pred_class == 1:
                    status_html = '<div class="badge-spam">SUSPICIOUS: SPAM PROMOTION</div>'
                else:
                    status_html = '<div class="badge-safe">SAFE: LEGITIMATE COMMUNICATIONS (HAM)</div>'
                
            with col_output:
                st.subheader("2. Threat Analysis Output")
                st.markdown(status_html, unsafe_allow_html=True)
                st.write("")
                
                # Probability Progress Breakdown
                pcol1, pcol2, pcol3 = st.columns(3)
                with pcol1:
                    st.metric("Ham Probability", f"{ham_prob*100:.1f}%")
                    st.progress(float(ham_prob))
                with pcol2:
                    st.metric("Spam Probability", f"{spam_prob*100:.1f}%")
                    st.progress(float(spam_prob))
                with pcol3:
                    st.metric("Phishing Probability", f"{phish_prob*100:.1f}%")
                    st.progress(float(phish_prob))

            with col_input:
                st.divider()
                # Highlighted Token Analysis
                st.markdown("**🔍 High-Risk Token Highlights:**")
                flagged_words = {"urgent", "verify", "account", "suspended", "password", "security", "credentials", "urltoken", "immediately", "bank"}
                words = user_input.split()
                highlighted_html = []
                for w in words:
                    w_str = str(w)
                    w_clean = re.sub(r'[^a-zA-Z0-9]', '', w_str).lower()
                    escaped_w = html.escape(w_str)
                    if "http" in w_str.lower() or "www." in w_str.lower() or w_clean == "urltoken":
                        highlighted_html.append(f'<span class="token-highlight-url">{escaped_w} (URL)</span>')
                    elif w_clean in flagged_words:
                        highlighted_html.append(f'<span class="token-highlight-critical">{escaped_w}</span>')
                    else:
                        highlighted_html.append(escaped_w)
                
                st.markdown(f'<div style="background-color: #F8F9FA; padding: 14px; border-radius: 8px; border: 1px solid #E2E8F0; line-height: 1.8;">{" ".join(highlighted_html)}</div>', unsafe_allow_html=True)

            with col_output:
                # Explainable AI (LIME) Section
                st.divider()
                with st.expander("📊 View LIME Token Importance Feature Contribution", expanded=True):
                    try:
                        explainer = ThreatExplainer(
                            gru_classifier.predict_proba if "GRU" in engine_choice else baseline_model.predict_proba
                        )
                        exp, _ = explainer.explain_instance(user_input, num_features=8)
                        
                        lime_list = exp.as_list()
                        tokens = [item[0] for item in lime_list]
                        scores = [item[1] for item in lime_list]
                        
                        fig, ax = plt.subplots(figsize=(8, 3.5))
                        colors = ['#D9381E' if s > 0 else '#10B981' for s in scores]
                        ax.barh(tokens, scores, color=colors)
                        ax.axvline(0, color='#64748B', linestyle='--', linewidth=0.8)
                        ax.set_title('LIME Feature Contribution Scores')
                        ax.set_xlabel('Contribution to Threat Class Probability')
                        st.pyplot(fig)
                    except Exception as e:
                        st.info(f"LIME visualization note: {e}")

# -----------------------------------------------------------------------------
# TAB 2: BATCH CSV PROCESSING
# -----------------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch Message Evaluation")
    st.markdown("Upload a CSV file containing a column named `text` or `message` for bulk threat scanning.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_df)}** records.")
            text_cols = [c for c in batch_df.columns if any(k in c.lower() for k in ['text', 'message', 'email', 'body'])]
            
            if text_cols:
                target_col = text_cols[0]
                if st.button("RUN BULK CLASSIFICATION", type="primary"):
                    with st.spinner("Processing batch records..."):
                        preds_probs = gru_classifier.predict_proba(batch_df[target_col].tolist()) if gru_classifier else baseline_model.predict_proba(batch_df[target_col].tolist())
                        pred_labels = np.argmax(preds_probs, axis=1)
                        
                        label_map = {0: 'Ham', 1: 'Spam', 2: 'Phishing'}
                        batch_df['Predicted_Threat'] = [label_map[p] for p in pred_labels]
                        batch_df['Phishing_Confidence'] = preds_probs[:, 2]
                        
                        st.dataframe(batch_df, use_container_width=True)
                        st.success("Batch threat classification completed!")
            else:
                st.error("No text/message column found in uploaded CSV.")
        except Exception as e:
            st.error(f"Error parsing CSV file: {e}")

# -----------------------------------------------------------------------------
# TAB 3: ARCHITECTURE & BENCHMARKS
# -----------------------------------------------------------------------------
with tab_analytics:
    st.subheader("📚 Literature-Backed Architecture & Benchmark Metrics")
    
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown("""
        ### Architectural Principles
        1. **3-Class Taxonomy**: Categorizes communications into Ham, Spam, and Phishing/Smishing.
        2. **URL Isolation**: Standardizes links to `<URL>` (`urltoken`) so NLP models scrutinize linguistic context rather than relying solely on domain reputation.
        3. **Sequence Length**: Fixed `maxlen=150` with `padding='post'`, `truncating='post'` balances SMS and Email payloads efficiently.
        """)
    with bcol2:
        st.markdown("""
        ### Performance Metrics (571k Unified Messages)
        - **Baseline Naive Bayes Accuracy**: **92.98%**
        - **GRU Deep Learning Accuracy**: **94.33%**
        - **Class Weights**:
          - Ham (0): `0.896`
          - Spam (1): `0.750`
          - Phishing (2): `1.815`
        """)

st.divider()
st.caption("IMPREX Mark3 Threat Intelligence System • Powered by PyTorch & Streamlit")
