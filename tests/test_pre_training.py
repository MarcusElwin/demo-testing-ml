import unittest
from testing_ml.data.dummy import create_dummy_data

NUM_FEATURES = ...

class TestPreTraining(unittest.TestCase):
    """Test class for pre-training tests"""
    
    def test_model_input(self):
        raise NotImplementedError
    
    def test_each_sequence_is_a_unique_user(self):
        expected_num_users = 100
        dataset = create_dummy_data(expected_num_users)
        actual_num_users = len(list(set(dataset["user_id"])))
        self.assertEqual(expected_num_users, actual_num_users)

    def test_scaling_of_dataset(self):
        raise NotImplementedError