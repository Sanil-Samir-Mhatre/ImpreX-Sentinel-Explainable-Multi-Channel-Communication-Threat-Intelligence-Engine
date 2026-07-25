"""
Streamlit Cloud Entrypoint for ImpreX Sentinel
"""

import sys
import os

model_dir = os.path.dirname(os.path.abspath(__file__))
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

from app_mark3 import *
