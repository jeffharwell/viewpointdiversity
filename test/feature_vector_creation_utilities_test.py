import unittest
import numpy as np
from src.viewpointdiversity.feature_vector_creation_utilities import create_has_sentiments_present_vector, create_word2vec_present_vector


class FeatureVectorCreationUtilitiesTest(unittest.TestCase):

    def test_create_has_sentiments_present_vector(self):
        """
        Test class to ensure that the function create_has_sentiments_present_vector is actually working.
        """
        v1 = [0.345, 0.655, 0, -0.3125, 0.6875, 0.345, 0.655, 0.0]
        v2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertTrue(create_has_sentiments_present_vector(v1, v2) == [1, 0])
        v1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v2 = [0, 0.7695000000000001, 0.3073333333333333, 0.5, 1.0, 0.0, 1.0, 0.375]
        self.assertTrue(create_has_sentiments_present_vector(v1, v2) == [0, 1])
        v1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.assertTrue(create_has_sentiments_present_vector(v1, v2) == [0, 0])
        v1 = [0.34, 0.66, 0, 0.2, 0.2, 0.34, 0.66, 0.0]
        v2 = [0.3415, 0.579, 0.159, 0.3125, 0.4625, 0.371, 0.629, 0.159]
        self.assertTrue(create_has_sentiments_present_vector(v1, v2) == [1, 1])

    def test_create_word2vec_present_vector(self):
        """
        Test class to ensure that the function create_has_sentiments_present_vector is actually working.
        """
        # v1 = [0.345, 0.655, 0, -0.3125, 0.6875, 0.345, 0.655, 0.0]
        # v2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v1 = np.random.rand(300)
        v2 = np.zeros(300)
        self.assertTrue(create_word2vec_present_vector(v1, v2) == [1, 0])
        v1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v2 = [0, 0.7695000000000001, 0.3073333333333333, 0.5, 1.0, 0.0, 1.0, 0.375]
        v1 = np.zeros(300)
        v2 = np.random.rand(300)
        self.assertTrue(create_word2vec_present_vector(v1, v2) == [0, 1])
        v1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v1 = np.zeros(300)
        v2 = np.zeros(300)
        self.assertTrue(create_word2vec_present_vector(v1, v2) == [0, 0])
        v1 = [0.34, 0.66, 0, 0.2, 0.2, 0.34, 0.66, 0.0]
        v2 = [0.3415, 0.579, 0.159, 0.3125, 0.4625, 0.371, 0.629, 0.159]
        v1 = np.random.rand(300)
        v2 = np.random.rand(300)
        self.assertTrue(create_word2vec_present_vector(v1, v2) == [1, 1])


if __name__ == '__main__':
    unittest.main()

