"""
Explainable AI (XAI) Module for Mark3 (LIME Explainer Integration)

Wraps model prediction into a clean pipeline accepting list[str] and outputting
a (N, 3) probability matrix required by LIME's LimeTextExplainer.
"""

from lime.lime_text import LimeTextExplainer
from preprocessing import clean_text

class ThreatExplainer:
    def __init__(self, model_predict_proba_fn, class_names=None):
        if class_names is None:
            class_names = ['Ham', 'Spam', 'Phishing']
        self.class_names = class_names
        self.model_predict_proba_fn = model_predict_proba_fn
        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def predict_pipeline(self, texts):
        """
        LIME Wrapper function:
        - Receives a list of raw text strings from LIME
        - Passes through preprocessing, tokenization, and padding pipeline
        - Returns (N, 3) float probability matrix
        """
        return self.model_predict_proba_fn(texts)

    def explain_instance(self, raw_text: str, num_features: int = 10, target_class: int = None):
        """
        Generates token importance explanation for a given message.
        """
        if target_class is None:
            # Predict default class
            probs = self.predict_pipeline([raw_text])[0]
            target_class = int(probs.argmax())

        exp = self.explainer.explain_instance(
            raw_text,
            self.predict_pipeline,
            num_features=num_features
        )
        return exp, target_class
