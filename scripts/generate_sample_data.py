"""
Generate Sample Data Entrypoint
"""

import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.data.sample.generate_sample import generate_sample_signals

if __name__ == "__main__":
    print("Generating VibraGuard sample vibration datasets...")
    generate_sample_signals()
