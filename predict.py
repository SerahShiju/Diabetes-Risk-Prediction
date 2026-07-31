"""
=========================================================
Diabetes Risk Prediction - Inference Engine
=========================================================

This script loads the trained deployment artifacts and
generates diabetes risk predictions for new patients.

Author : Serah Ann Shiju
Project: Diabetes Risk Prediction with a Calibrated
         Stacking Ensemble
=========================================================
"""

import os
import json
import joblib
import argparse
import pandas as pd
import numpy as np
# =========================================================
# Project Paths
# =========================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")

# =========================================================
# Model File Paths
# =========================================================

PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

XGB_PATH = os.path.join(MODELS_DIR, "xgb.pkl")

LGBM_PATH = os.path.join(MODELS_DIR, "lightgbm.pkl")

MLP_PATH = os.path.join(MODELS_DIR, "mlp.pkl")

META_MODEL_PATH = os.path.join(MODELS_DIR, "meta.pkl")

CALIBRATED_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "calibrated.pkl"
)

THRESHOLD_PATH = os.path.join(MODELS_DIR, "threshold.pkl")

FEATURE_NAMES_PATH = os.path.join(
    MODELS_DIR,
    "feature_names.pkl"
)

METADATA_PATH = os.path.join(
    MODELS_DIR,
    "metadata.json"
)

# =========================================================
# Inference Pipeline
# =========================================================

class InferencePipeline:
    """
    End-to-end inference pipeline for diabetes risk prediction.
    """

    def __init__(self):

        print("Loading deployment artifacts...")

        # Load preprocessing pipeline
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)

        # Load base models
        self.xgb = joblib.load(XGB_PATH)
        self.lightgbm = joblib.load(LGBM_PATH)
        self.mlp = joblib.load(MLP_PATH)

        # Load meta learner
        self.meta_model = joblib.load(META_MODEL_PATH)

        # Load calibrated model
        self.calibrated_model = joblib.load(CALIBRATED_MODEL_PATH)

        # Load threshold
        self.threshold = joblib.load(THRESHOLD_PATH)

        # Load feature names
        self.feature_names = joblib.load(FEATURE_NAMES_PATH)

        # Load target labels
        self.target_labels = joblib.load(
            os.path.join(MODELS_DIR, "target_labels.pkl")
        )

        # Load metadata
        with open(METADATA_PATH, "r") as file:
            self.metadata = json.load(file)

        print("Deployment artifacts loaded successfully.\n")
        # =========================================================
    # Data Preprocessing
    # =========================================================
    def preprocess(self, input_data):
        """
        Preprocess raw patient data.
        """

        # Expected raw input columns
        expected_columns = list(self.preprocessor.feature_names_in_)

        # Check for missing columns
        missing_columns = set(expected_columns) - set(input_data.columns)

        if missing_columns:
            raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}")

        # Reorder columns exactly as used during training
        input_data = input_data[expected_columns]
        # Transform data
        transformed_data = self.preprocessor.transform(input_data)

        # Convert to DataFrame with feature names
        transformed_data = pd.DataFrame(
        transformed_data,
        columns=self.feature_names,
        index=input_data.index)

        return transformed_data   

    # =========================================================
    # Probability Prediction
    # =========================================================
    def predict_proba(self, input_data):
        """
        Generate calibrated diabetes risk probabilities.

        Parameters
        ----------
        input_data : pandas.DataFrame

        Returns
        -------
        numpy.ndarray
            Calibrated probability for the positive class.
        """

        # -----------------------------------------------------
        # Preprocess input
        # -----------------------------------------------------
        X = self.preprocess(input_data)

        # -----------------------------------------------------
        # Base model probabilities
        # -----------------------------------------------------
        xgb_prob = self.xgb.predict_proba(X)[:, 1]

        lgbm_prob = self.lightgbm.predict_proba(X)[:, 1]

        mlp_prob = self.mlp.predict_proba(X.to_numpy())[:, 1]

        # -----------------------------------------------------
        # Create meta-features
        # -----------------------------------------------------
        meta_features = pd.DataFrame(
            {
                "XGB": xgb_prob,
                "LGBM": lgbm_prob,
                "MLP": mlp_prob
            }
        )

        # -----------------------------------------------------
        # Meta-model probability
        # -----------------------------------------------------
        # Meta-model probability (optional, mainly for debugging)
        meta_probability = self.meta_model.predict_proba(meta_features)[:, 1]

        # -----------------------------------------------------
        # Probability Calibration
        # -----------------------------------------------------
        calibrated_probability = self.calibrated_model.predict_proba(
            meta_features
        )[:, 1]

        return calibrated_probability
        # =========================================================
    # Final Prediction
    # =========================================================
    def predict(self, input_data):
        """
        Generate diabetes risk predictions.

        Parameters
        ----------
        input_data : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
            Prediction results.
        """

        # Get calibrated probabilities
        probabilities = self.predict_proba(input_data)

        # Apply decision threshold
        predictions = (
            probabilities >= self.threshold
        ).astype(int)

        # Convert to readable labels
        labels = [
            self.target_labels[p]
            for p in predictions
        ]

        # Build output dataframe
        results = pd.DataFrame({
            "Probability": probabilities,
            "Threshold": self.threshold,
            "Prediction": predictions,
            "Risk_Label": labels
        })

        return results
        # =========================================================
    # Batch Prediction
    # =========================================================
    def batch_predict(self, csv_path):
        """
        Predict diabetes risk for all patients in a CSV file.

        Parameters
        ----------
        csv_path : str
            Path to the input CSV file.

        Returns
        -------
        pandas.DataFrame
            Prediction results for all patients.
        """

        # Load input data
        input_data = pd.read_csv(csv_path)

        # Generate predictions
        results = self.predict(input_data)

        return results
# =========================================================
# Command Line Interface
# =========================================================

# =========================================================
# Command Line Interface
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Diabetes Risk Prediction Inference Engine"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV file to save predictions"
    )

    args = parser.parse_args()

    try:

        print("=" * 70)
        print("Diabetes Risk Prediction Inference Engine")
        print("=" * 70)

        pipeline = InferencePipeline()

        results = pipeline.batch_predict(args.input)

        print("\nPrediction Results\n")

        for i, row in results.iterrows():

            print("=" * 50)
            print(f"Patient #{i + 1}")
            print("=" * 50)

            print(f"Probability : {row['Probability'] * 100:.2f}%")
            print(f"Threshold   : {pipeline.threshold * 100:.2f}%")
            print(f"Prediction  : {row['Risk_Label']}")

            if row["Prediction"] == 1:
                print("Decision    : Above Threshold")
            else:
                print("Decision    : Below Threshold")

            print()

        if args.output:

            results.to_csv(args.output, index=False)

            print(f"Predictions saved to: {args.output}")

    except Exception as e:

        print("\nERROR")
        print("-" * 40)
        print(str(e))


if __name__ == "__main__":
    main()
