"""
Executable wrapper for Step 1: Preprocessing
"""
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess_pipeline
from src.utils import ExecutionTimer

if __name__ == "__main__":
    with ExecutionTimer("Step 1: Preprocessing (data/raw -> data/processed)"):
        preprocess_pipeline()
