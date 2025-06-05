import unittest
import os
import sys
import io
import numpy as np
import torch

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from app
from app import ModelCache, PredictionResponse

class TestBasicFunctionality(unittest.TestCase):
    def test_model_loading(self):
        """Test that the model can be loaded without errors"""
        # This is an async function, so we need to run it in an event loop
        import asyncio
        model, feature_extractor = asyncio.run(ModelCache.get_model())
        self.assertIsNotNone(model)
        self.assertIsNotNone(feature_extractor)
    
    def test_prediction_response(self):
        """Test that the prediction response can be created correctly"""
        response = PredictionResponse(
            emotion="bahagia",
            confidence=0.85,
            probabilities={"marah": 0.05, "jijik": 0.02, "takut": 0.03, 
                           "bahagia": 0.85, "netral": 0.03, "sedih": 0.02},
            file_id="test-id",
            storage_path="test-path"
        )
        self.assertEqual(response.emotion, "bahagia")
        self.assertEqual(response.confidence, 0.85)
        self.assertEqual(response.file_id, "test-id")

if __name__ == "__main__":
    unittest.main()
