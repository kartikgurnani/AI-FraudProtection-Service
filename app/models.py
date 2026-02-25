"""
models.py

Unified model loader for AI-FraudProtection-Service.
Loads all ML models into memory once and exposes them globally.
"""

import joblib
from pathlib import Path


class FraudModels:
    """Singleton-style model loader."""

    _instance = None

    def __new__(cls, model_dir: str = "models"):
        if cls._instance is None:
            cls._instance = super(FraudModels, cls).__new__(cls)
            cls._instance._load_models(model_dir)
        return cls._instance

    def _load_models(self, model_dir: str):
        model_dir = Path(model_dir)

        print("Loading fraud detection models...")

        self.pca = joblib.load(model_dir / "pca_model.pkl")
        self.isolation_forest = joblib.load(model_dir / "isolation_forest_model.pkl")
        self.one_class_svm = joblib.load(model_dir / "one_class_svm_model.pkl")
        self.local_outlier_factor = joblib.load(model_dir / "local_outlier_factor_model.pkl")
        self.elliptic_envelope = joblib.load(model_dir / "elliptic_envelope_model.pkl")
        self.som = joblib.load(model_dir / "som_model.pkl")
        self.tsne = joblib.load(model_dir / "tsne_model.pkl")

        print("All models loaded successfully.")


# global instance (recommended for APIs)
fraud_models = FraudModels()
