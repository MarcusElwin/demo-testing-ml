import unittest
import numpy as np
import pandas as pd
from testing_ml.data.dummy import create_dummy_data, get_training_data
from testing_ml.pipeline.model import get_model_pipeline
from sklearn.model_selection import GroupShuffleSplit

NUM_FEATURES = 150
NUM_SAMPLES = 100

def train_test_split_per_column(df: pd.DataFrame, col="user_id", test_size=0.2, n_splits=2, seed=123):
    """Split data into train and test set based on col, rovides randomized train/test
    indices to split data according to a third-party provided group.
    """
    train_inds, test_inds = next(
        GroupShuffleSplit(
            test_size=test_size, n_splits=n_splits, random_state=seed
        ).split(df, groups=df[col])
    )

    return df.iloc[train_inds], df.iloc[test_inds]

class TestPreTraining(unittest.TestCase):
    """Test class for pre-training tests"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dummy_data = create_dummy_data(num_rows=1000)
        cls.features, cls.target = get_training_data(num_rows=100)
        cls.model = get_model_pipeline().fit(cls.features.reshape(1, -1), cls.target.reshape(1, -1))
    
    def test_model_input(self):
        features, _ = get_training_data(num_rows=100)
        self.assertEqual(features.shape[0], NUM_FEATURES)
        self.assertEqual(features.shape[1], NUM_SAMPLES)
    
    def test_each_sequence_is_a_unique_user(self):
        expected_num_users = 100
        dataset = create_dummy_data(expected_num_users)
        actual_num_users = len(list(set(dataset["user_id"])))
        self.assertEqual(expected_num_users, actual_num_users)

    def test_scaling_of_dataset(self):
        num_users = 100
        features, _ = get_training_data(num_users)
        # extract features and create concat array

        for input in features:
            # get max and min
            max_value = np.max(input)
            min_value = np.min(input)

            # test scale of data
            self.assertGreaterEqual(min_value, 0.0)
            self.assertLessEqual(max_value, 1.0)

            # check that no value is outside of the bounds -1.5,1.5
            self.assertTrue(np.any(input > 0.0))
            self.assertTrue(np.any(input < 1.0))

    def test_no_user_leakage_all_sets_data_split(self):
        # split into training, test and validation
        train, test = train_test_split_per_column(self.dummy_data)

        # check intersection
        intersection_train_test = set(
            train["user_id"]
        ).intersection(set(test["user_id"]))

        # ensure that we are not leaking
        empty_set = set()
        self.assertEqual(intersection_train_test, empty_set)
