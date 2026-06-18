import unittest
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent/"src"))
from preprocessing import(build_pipeline,load_data,FEATURE_COLS,CONTINUOUS_COLS,VALID_RANGES,validate_input_ranges)

Sample_Data=pd.DataFrame({
    "age": [45, 54, 63, 39, 58],
    "trestbps": [120, 140, 130, 110, 150],
    "chol": [220, 250, np.nan, 180, 310],
    "thalach": [170, 150, 145, 185, 132],
    "oldpeak": [0.0, 1.2, 2.3, 0.4, np.nan],
    "sex": [1, 0, 1, 0, 1],
    "cp": [3, 2, 4, 1, 2],
    "fbs": [0, 1, 0, 0, 1],
    "restecg": [0, 1, 2, 0, 1],
    "exang": [0, 1, 1, 0, 1],
})

Sample_Labels=np.array([0,1,0,0,1])

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline=build_pipeline()
        cls.X_Transform=cls.pipeline.fit_transform(Sample_Data)

        model_path=Path(__file__).resolve().parent.parent/"models"/"best_model.pkl"
        cls.model=joblib.load(model_path) if model_path.exists() else None

    def test_prediction_shape(self):
        if self.model is None:
            self.skipTest("best_model.pkl not found")

        predictions=self.model.predict(Sample_Data[FEATURE_COLS])
        self.assertEqual(predictions.shape, (len(Sample_Data),))

    def test_prediction_probability_shape(self):
        if self.model is None:
            self.skipTest("best_model.pkl not found")

        probabilities=self.model.predict_proba(Sample_Data[FEATURE_COLS])
        self.assertEqual(probabilities.shape, (len(Sample_Data), 2))

    def test_prediction_probability_range(self):
        if self.model is None:
            self.skipTest("best_model.pkl not found")

        probabilities=self.model.predict_proba(Sample_Data[FEATURE_COLS])
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))

    def test_prediction_probability_sum(self):
        if self.model is None:
            self.skipTest("best_model.pkl not found")

        probabilities=self.model.predict_proba(Sample_Data[FEATURE_COLS])
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))

    def test_validate_input_ranges(self):
        valid_violations=validate_input_ranges(Sample_Data[FEATURE_COLS])
        self.assertEqual(valid_violations, [])

        invalid_sample=Sample_Data.copy()
        invalid_sample.loc[0, "age"]=VALID_RANGES["age"][1]+1
        invalid_sample.loc[1, "chol"]=VALID_RANGES["chol"][0]-1

        invalid_violations=validate_input_ranges(invalid_sample[FEATURE_COLS])
        self.assertIn("age", invalid_violations)
        self.assertIn("chol", invalid_violations)

    def test_pipeline_deterministic_output(self):
        transformed_once=self.pipeline.transform(Sample_Data[FEATURE_COLS])

        second_pipeline=build_pipeline()
        transformed_twice=second_pipeline.fit_transform(Sample_Data[FEATURE_COLS])

        np.testing.assert_allclose(self.X_Transform, transformed_once)
        np.testing.assert_allclose(self.X_Transform, transformed_twice)

    def test_no_missing_values_after_preprocessing(self):
        transformed=self.pipeline.transform(Sample_Data[FEATURE_COLS])
        self.assertFalse(np.isnan(transformed).any())

    def test_preprocessing_output_shape(self):
        transformed=self.pipeline.transform(Sample_Data[FEATURE_COLS])
        self.assertEqual(transformed.shape, (len(Sample_Data), len(FEATURE_COLS)))

if __name__=="__main__":
    unittest.main(verbosity=2)
