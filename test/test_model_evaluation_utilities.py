import unittest
import numpy as np
from src.viewpointdiversitydetection.model_evaluation_utilities import generate_markdown_table


class ModelEvaluationUtilitiesTest(unittest.TestCase):

    def test_generate_markdown_table(self):
        parameters = {'C': 5, 'gamma': .0001, 'class w': 'balanced',
                      'IAA': 0.2}
        answers = np.array(['b', 'b', 'b', 'b', 'a', 'b', 'a'])
        predictions = np.array(['a', 'a', 'b', 'b', 'a', 'a', 'a'])
        probabilities_list = [[.9, .1], [.4, .6], [.2, .8], [.3, .7], [.8, .2], [.55, .45], [.95, .05]]
        probabilities = np.array([np.array(i) for i in probabilities_list])
        top_number = 3
        label_a = 'a'
        label_b = 'b'
        corpus_name = 'Testing'
        search_terms = ['term 1', 'term 2']

        t = generate_markdown_table(corpus_name, search_terms, parameters, answers, predictions, probabilities,
                                    top_number, label_a, label_b)
        print(t)
        self.assertTrue(t)


if __name__ == '__main__':
    unittest.main()