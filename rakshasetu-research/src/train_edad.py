"""
Executable entrypoint for Step 3: train_edad.py
"""
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import train_edad_pipeline
from src.utils import ExecutionTimer

if __name__ == "__main__":
    with ExecutionTimer("Step 3: EDAD Model Training & Checkpointing"):
        train_edad_pipeline()
