"""
Automated Unit and Integration Tests for Customer Churn Pipeline.
Uses Python's standard `unittest` library (zero external test dependencies).
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import joblib

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_preprocessing import clean_churn_data, build_preprocessor, NUMERICAL_COLS, CATEGORICAL_COLS


class TestChurnPipeline(unittest.TestCase):

    def test_clean_churn_data_whitespace_handling(self):
        """Tests that whitespace in TotalCharges is correctly converted to 0.0 float."""
        raw_sample = pd.DataFrame({
            "customerID": ["TEST_1", "TEST_2"],
            "tenure": [0, 5],
            "MonthlyCharges": [20.0, 70.0],
            "TotalCharges": [" ", " 350.50 "],
            "Churn": ["No", "Yes"]
        })
        cleaned = clean_churn_data(raw_sample)
        self.assertEqual(cleaned["TotalCharges"].iloc[0], 0.0)
        self.assertEqual(cleaned["TotalCharges"].iloc[1], 350.50)
        self.assertEqual(cleaned["Churn"].iloc[0], 0)
        self.assertEqual(cleaned["Churn"].iloc[1], 1)

    def test_preprocessor_column_output(self):
        """Tests that ColumnTransformer executes without errors on representative data."""
        sample_df = pd.DataFrame([{
            "customerID": "TEST_001",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 45.0,
            "TotalCharges": 540.0
        }])

        preprocessor = build_preprocessor()
        transformed = preprocessor.fit_transform(sample_df)
        self.assertIsInstance(transformed, np.ndarray)
        self.assertEqual(transformed.shape[0], 1)
        self.assertGreater(transformed.shape[1], 0)

    def test_champion_model_prediction(self):
        """Tests that the trained pipeline loads and produces valid probabilities."""
        model_path = os.path.join("models", "best_model.joblib")
        self.assertTrue(os.path.exists(model_path), "Champion model file does not exist.")

        pipeline = joblib.load(model_path)
        sample_df = pd.DataFrame([{
            "customerID": "TEST_002",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 24,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 95.5,
            "TotalCharges": 2292.0
        }])

        proba = pipeline.predict_proba(sample_df)
        self.assertEqual(proba.shape, (1, 2))
        self.assertTrue(0.0 <= proba[0][0] <= 1.0)
        self.assertTrue(0.0 <= proba[0][1] <= 1.0)
        self.assertTrue(np.isclose(proba[0][0] + proba[0][1], 1.0))

        pred = pipeline.predict(sample_df)
        self.assertIn(pred[0], [0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
