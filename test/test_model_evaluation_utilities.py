import unittest
from operator import itemgetter

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

    def test_create_run_stats_lift(self):
        """
        Verifies the top and bottom lift calculation is being done properly.
        """
        answers = np.array(['b', 'b', 'b', 'b', 'a', 'b', 'a'])
        predictions = np.array(['a', 'a', 'b', 'b', 'a', 'a', 'a'])
        probabilities_list = [[.9, .1], [.4, .6], [.2, .8], [.3, .7], [.8, .2], [.55, .45], [.95, .05]]
        probabilities = np.array([np.array(i) for i in probabilities_list])
        top_number = 3
        label_a = 'a'
        label_b = 'b'

        # Unpack our probabilities
        prob1 = [p[0] for p in probabilities]
        prob2 = [p[1] for p in probabilities]
        # Create a new data structure so that we can sort
        all_data = list(zip(answers, predictions, prob1, prob2))
        # Sort by probability one, this will correspond to our first class, stance 0, oppose
        sorted_data_zeros_first = sorted(all_data, key=itemgetter(2),
                                         reverse=True)  # Highest prob of getting a 0 is sorted first\
        reversed_data = all_data.copy()
        reversed_data.reverse()
        # Sort by probability two, this will correspond to our second class, stance 1, support
        sorted_data_ones_first = sorted(reversed_data, key=itemgetter(3),
                                        reverse=True)  # Highest prob of getting a 1 is at the beginning
        # Class 1 - Slice of top performers
        answers_class_1 = [a[0] for a in sorted_data_zeros_first[0:top_number]]
        predictions_class_1 = [a[1] for a in sorted_data_zeros_first[0:top_number]]
        self.assertEqual(answers_class_1, ['a', 'b', 'a'])
        self.assertEqual(predictions_class_1, ['a', 'a', 'a'])

        # Class 2 - Slice of top performers
        answers_class_2 = [a[0] for a in sorted_data_ones_first[0:top_number]]
        predictions_class_2 = [a[1] for a in sorted_data_ones_first[0:top_number]]
        self.assertEqual(answers_class_2, ['b', 'b', 'b'])
        self.assertEqual(predictions_class_2, ['b', 'b', 'a'])

        # lift for class 1 is calculated as the percent of class 1 in the top predictions divided by the
        # percent of class 1 overall
        # lift for class 2 is calculated as the percent of class 2 in the bottom predictions divided by the
        # percent of class 2 overall

        # get the total percent of class 1 and class 2 for the entire dataset
        total_percent_class_1 = len([x for x in answers if x == 'a'])/len(answers)
        total_percent_class_2 = len([x for x in answers if x == 'b'])/len(answers)

        # get the percentage of class 1 and class 2 in our top and bottom strongest predictions respectively
        top_percent_class_1 = len([x for x in answers_class_1 if x == 'a'])/len(answers_class_1)
        top_percent_class_2 = len([x for x in answers_class_2 if x == 'b'])/len(answers_class_2)

        # Now calculate lift for class 1 from the top predictions and class 2 from the bottom predictions
        lift_class_1 = top_percent_class_1 / total_percent_class_1
        lift_class_2 = top_percent_class_2 / total_percent_class_2

        """
        print(
            f"Class 1 (a): top percent {top_percent_class_1:.2f}, "
            f"total_percent {total_percent_class_1:.2f}, lift {lift_class_1:.2f}")
        print(
            f"Class 2 (b): top percent {top_percent_class_2:.2f}, "
            f"total_percent {total_percent_class_2:.2f}, lift {lift_class_2:.2f}")
        """

        # answers, predictions, probabilities, top_number, class_1_label, class_2_label
        t = create_run_stats(answers, predictions, probabilities, top_number, label_a, label_b)
        self.assertEqual(t['Lift']['a'], lift_class_1)
        self.assertEqual(t['Lift']['b'], lift_class_2)


if __name__ == '__main__':
    unittest.main()
