import unittest
import numpy as np
from src.viewpointdiversitydetection.model_evaluation_utilities import generate_markdown_table
from src.viewpointdiversitydetection.model_evaluation_utilities import create_run_stats
from src.viewpointdiversitydetection.model_evaluation_utilities import analyze_top_predictions


class ModelEvaluationUtilitiesTest(unittest.TestCase):

    def test_generate_markdown_table(self):
        """
        Again, not much of a test class. Asserts that the markdown generation class will return a string and not
        raise any errors.
        """
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

    def test_print_combined_stats(self):
        """
        Not much of a test class, basically asserts that the function that prints out the statistics
        doesn't raise an exception.
        """
        answers = np.array(['b', 'b', 'b', 'b', 'a', 'b', 'a'])
        predictions = np.array(['a', 'a', 'b', 'b', 'a', 'a', 'a'])
        probabilities_list = [[.9, .1], [.4, .6], [.2, .8], [.3, .7], [.8, .2], [.55, .45], [.95, .05]]
        probabilities = np.array([np.array(i) for i in probabilities_list])
        top_number = 3
        label_a = 'a'
        label_b = 'b'

        analyze_top_predictions(answers, predictions, probabilities,
                                top_number, label_a, label_b)

    def test_create_run_stats(self):
        """
        Nothing to exciting, it doesn't verify the calculations, just ensures that the method returns a few key
        points of data and doesn't raise any exceptions.
        """
        answers = np.array(['b', 'b', 'b', 'b', 'a', 'b', 'a'])
        predictions = np.array(['a', 'a', 'b', 'b', 'a', 'a', 'a'])
        probabilities_list = [[.9, .1], [.4, .6], [.2, .8], [.3, .7], [.8, .2], [.55, .45], [.95, .05]]
        probabilities = np.array([np.array(i) for i in probabilities_list])
        top_number = 3
        label_a = 'a'
        label_b = 'b'

        # answers, predictions, probabilities, top_number, class_1_label, class_2_label
        t = create_run_stats(answers, predictions, probabilities, top_number, label_a, label_b)
        print(t)
        self.assertEqual(set(list(t.keys())), {'TPR', 'TNR', 'PPV', 'NPV', 'Bal Acc', 'Lift'})


if __name__ == '__main__':
    unittest.main()
