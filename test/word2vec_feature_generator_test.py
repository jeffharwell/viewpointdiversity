import unittest
import numpy as np

from src.viewpointdiversitydetection import Word2VecFeatureGenerator

class Word2VecFeatureGeneratorTest(unittest.TestCase):

    def test_avg_vector_no_divide_by_zero(self):
        """
        When all of the word2vec features are empty we shouldn't hit a divide-by-zero
        RuntimeError if we are not including zeros in the averages.

        :return:
        """
        w2v_obj = Word2VecFeatureGenerator()
        zero_vector = np.zeros(w2v_obj.vector_size)
        vectors = [zero_vector.copy(), zero_vector.copy(), zero_vector.copy()]
        avg = w2v_obj._average_vector(vectors)
        self.assertTrue(np.array_equal(avg, zero_vector))

        w2v_obj.include_zeros_in_averages = False

        avg = w2v_obj._average_vector(vectors)
        self.assertTrue(np.array_equal(avg, zero_vector))

if __name__ == '__main__':
    unittest.main()