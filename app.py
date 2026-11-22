"""
Root Entrypoint for Streamlit Cloud and Container Deployment.
Runs the DocuAI Interactive Dashboard.
"""
import sys
import os

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Run the Streamlit application
from src.ui.app import *
