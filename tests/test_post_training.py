import unittest
import numpy as np
import pandas as pd
from testing_ml.data.dummy import get_training_data
from testing_ml.pipeline.model import get_model_pipeline

class TestPostTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.features, cls.target = get_training_data(num_rows=100)
        cls.model = get_model_pipeline().fit(cls.features.reshape(1, -1), cls.target.reshape(1, -1))

    def test_feature_invariance(self):
        # get slightly different data
        num_rows = 100
        features, targets = get_training_data(num_rows)
        features_inv, _ = get_training_data(num_rows, invariance = True)
        # get predictions
        predictions = self.model.predict(features.reshape(1, -1))
        predictions_inv = self.model.predict(features_inv.reshape(1, -1))
        # check how many correct
        num_correct_before = ((targets == predictions).sum())
        num_correct_after = ((targets == predictions_inv).sum())

        self.assertAlmostEqual(
            num_correct_before,
            num_correct_after,
            msg="Different predictions!",
            delta=0.95 * num_rows
        )