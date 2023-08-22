import unittest
import numpy as np
import pandas as pd
from testing_ml.data.dummy import get_training_data
from testing_ml.pipeline.model import get_model_pipeline
from scipy import stats

def is_drifting_distribution(dist_a: np.array, dist_b: np.array, null_hypothesis_threshold: float = 0.05):
    """Performs Kolmogorov-Smirnov-test on the amount distribution
    HO: distributions are the same, reject the null hypothesis in favor of the alternative
    if the p-value is less than null_hypothesis_threshold
    """
    test = stats.ks_2samp(dist_a, dist_b)
    return (
        test.pvalue < null_hypothesis_threshold, 
        test.pvalue, 
        test.statistic
    )

class TestDataDrift(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.features, cls.target = get_training_data(num_rows=100)
        cls.model = get_model_pipeline().fit(cls.features.reshape(1, -1), cls.target.reshape(1, -1))

    def test_is_same_distribution(self):
        dist_a = self.features[0]
        dist_b = self.features[0]
        check, _, _ = is_drifting_distribution(dist_a, dist_b)
        self.assertFalse(check)
        
    
    def test_is_different_distribution(self):
        dist_a = self.features[-1]
        dist_b = self.features[0]
        check, _, _ = is_drifting_distribution(dist_a, dist_b)
        self.assertTrue(check)