"""
RakshaSetu Anomaly Detection Research Pipeline - Utility Module
Synthetic Anomaly Benchmark Utilities
"""

import os
import random
import time
import yaml
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Sets deterministic seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device() -> torch.device:
    """Auto-detects CPU/GPU and returns torch.device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

class ExecutionTimer:
    """Context manager for timing pipeline steps."""
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.elapsed_sec = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START: {self.step_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_sec = time.perf_counter() - self.start_time
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DONE: {self.step_name} in {self.elapsed_sec:.3f}s")
