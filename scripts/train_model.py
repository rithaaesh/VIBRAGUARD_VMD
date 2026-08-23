"""
VibraGuard Model Training Script
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.training.trainer import ModelTrainer

if __name__ == "__main__":
    print("Executing VibraGuard Model Training Pipeline...")
    trainer = ModelTrainer(model_dir="models")
    meta = trainer.train_and_evaluate("Random Forest")
    print(f"Training Complete! Model Accuracy: {meta['accuracy']*100:.2f}%, F1-Score: {meta['f1_score']:.4f}")
