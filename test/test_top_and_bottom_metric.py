import math
import unittest
import numpy as np
from sklearn.dummy import DummyClassifier

from src.viewpointdiversitydetection import TopAndBottomMetric


class TopAndBottomMetricTest(unittest.TestCase):

    #
    # Test the Internals
    #

    def test_percent_must_be_less_than_half(self):
        """
        The class should throw a value error if you instruct it to consider more
        than half of the sample and the top n percent. If the percentage is over .5
        you would actually be evaluating some data elements more than once.
        """
        with self.assertRaises(ValueError):
            TopAndBottomMetric(.6)

    def test_calculate_confusion_matrix(self):
        """
        Ensure calculation is correct.
        """
        answers = np.array(['a', 'a', 'b', 'b', 'a', 'b', 'a', 'a', 'a', 'a'])
        predictions = np.array(['a', 'b', 'b', 'b', 'a', 'a', 'a', 'b', 'a', 'b'])
        m = TopAndBottomMetric(.1)

        tp, fp, tn, fn = m._calculate_confusion_matrix(answers, predictions, 'a')
        self.assertTrue(tp == 4)
        self.assertTrue(fp == 1)
        self.assertTrue(tn == 2)
        self.assertTrue(fn == 3)
        tp, fp, tn, fn = m._calculate_confusion_matrix(answers, predictions, 'b')
        self.assertTrue(tp == 2)
        self.assertTrue(fp == 3)
        self.assertTrue(tn == 4)
        self.assertTrue(fn == 1)

    def test_get_top_n_most_probable(self):
        answers = np.array(['b', 'a', 'b', 'b', 'a', 'b', 'a'])
        predictions = np.array(['a', 'b', 'b', 'b', 'a', 'a', 'a'])
        probabilities_list = [[.9, .1], [.4, .6], [.2, .8], [.3, .7], [.8, .2], [.55, .45], [.95, .05]]
        probabilities = np.array([np.array(i) for i in probabilities_list])

        # Unpack the probabilities
        prob_1 = [p[0] for p in probabilities]
        prob_2 = [p[1] for p in probabilities]

        top_bottom_percent = .4

        # create our object
        m = TopAndBottomMetric(top_bottom_percent)

        # calculate our number of samples (should be 3 based on this test data)
        top_n = round(top_bottom_percent * len(answers))
        self.assertTrue(top_n == 3)

        # Use the method to get our top_n most probably predictions and answers for each class
        a1, p1 = m._get_top_n_most_probable(top_n, answers, predictions, prob_1)
        a2, p2 = m._get_top_n_most_probable(top_n, answers, predictions, prob_2)

        # did we get the correct answers?
        self.assertTrue(p1 == ['a', 'a', 'a'])
        self.assertTrue(a1 == ['a', 'b', 'a'])
        self.assertTrue(p2 == ['b', 'b', 'b'])
        self.assertTrue(a2 == ['b', 'b', 'a'])

    def test_get_top_and_bottom_precision(self):
        # answers = np.array(['b', 'b', 'b', 'b', 'a', 'b', 'a'])
        # predictions = np.array(['a', 'a', 'b', 'b', 'a', 'a', 'a'])
        # probabilities_list = [[.9, .1], [.4, .6], [.2, .8], [.3, .7], [.8, .2], [.55, .45], [.95, .05]]
        # probabilities = np.array([np.array(i) for i in probabilities_list])

        top_a1 = ['a', 'b', 'a']  # tp = 2, fp = 1
        top_p1 = ['a', 'a', 'a']
        top_a2 = ['b', 'b', 'b']  # tp = 2, fp = 0
        top_p2 = ['b', 'b', 'a']

        top_bottom_percent = .4
        m = TopAndBottomMetric(top_bottom_percent)

        tp, fp, tn, fn = m._calculate_confusion_matrix(top_a1, top_p1, 'a')
        t = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
        tp, fp, tn, fn = m._calculate_confusion_matrix(top_a2, top_p2, 'b')
        b = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

        top_precision = m._calc_precision(t['tp'], t['fp'])
        bottom_precision = m._calc_precision(b['tp'], b['fp'])

        self.assertTrue(top_precision == (2 / (2 + 1)))  # tp / (tp + fp) 'a' is positive
        self.assertTrue(bottom_precision == (2 / (2 + 0)))  # tp / tp + fp 'b' is positive

    def test_get_from_estimator(self):
        data = np.array([1, 2, 1, 1, 2, 1, 2])
        answers = np.array(['b', 'b', 'b', 'b', 'b', 'b', 'a'])
        estimator = DummyClassifier(strategy="most_frequent")
        estimator.fit(data, answers)

        class_1, class_2 = estimator.classes_

        top_bottom_percent = .4
        m = TopAndBottomMetric(top_bottom_percent)
        n = 3
        top_cm, bottom_cm = m._get_top_bottom_confusion_matrix_from_estimator(n, estimator, data, answers)

        # the estimator will always guess 'b' with a probability of [0, 1], and since the probabilities
        # are all the same the sort algorithm will not change the order of the answers. But the metric flips the
        # arrays for class 2, so the top is the first three items, and the bottom is the last 3 items.
        # So the for the top confusion matrix, which will be positive for class 1, 'a', tn == 3. And for the
        # bottom confusion matrix, which will be positive for class 2, 'b', tp == 2.
        if class_1 == 'a':
            self.assertTrue(top_cm['tn'] == 3)
            self.assertTrue(bottom_cm['tp'] == 2)
        elif class_1 == 'b':
            self.assertTrue(top_cm['tp'] == 3)
            self.assertTrue(bottom_cm['tn'] == 2)

    #
    # Test the Metrics
    #

    def test_precision_metric(self):
        # answers = np.array(['b', 'b', 'b', 'b', 'a', 'b', 'a'])
        # predictions = np.array(['a', 'a', 'b', 'b', 'a', 'a', 'a'])
        # probabilities_list = [[.9, .1], [.4, .6], [.2, .8], [.3, .7], [.8, .2], [.55, .45], [.95, .05]]
        # probabilities = np.array([np.array(i) for i in probabilities_list])

        # top_a1 = ['a', 'b', 'a']  # tp = 2, fp = 1
        # top_p1 = ['a', 'a', 'a']
        # top_a2 = ['b', 'b', 'b']  # tp = 2, fp = 0
        # top_p2 = ['b', 'b', 'a']

        t = {'tp': 2, 'fp': 1}
        b = {'tp': 2, 'fp': 0}
        top_precision = (t['tp'] / (t['tp'] + t['fp']))
        bottom_precision = (b['tp'] / (b['tp'] + b['fp']))

        log_top_metric = math.log(top_precision + 1)
        log_bottom_metric = math.log(bottom_precision + 1)
        imbalance_penalty = math.log(abs(t['tp'] - b['tp']) + 1)
        metric_value = log_top_metric + log_bottom_metric - imbalance_penalty

        top_bottom_percent = .4
        m = TopAndBottomMetric(top_bottom_percent)
        self.assertTrue(m._calc_precision_metric(t, b) == metric_value)

    def test_balanced_accuracy_metric(self):
        # Test Data
        # answers = np.array(['b', 'b', 'b', 'b', 'a', 'b', 'a'])
        # predictions = np.array(['a', 'a', 'b', 'b', 'a', 'a', 'a'])
        # probabilities_list = [[.9, .1], [.4, .6], [.2, .8], [.3, .7], [.8, .2], [.55, .45], [.95, .05]]
        # probabilities = np.array([np.array(i) for i in probabilities_list])

        # Prepare the test data, create our top and bottom most probable confusion
        # matrices by hand from the above data
        # top_a1 = ['a', 'b', 'a']  # tp = 2, fp = 1
        # top_p1 = ['a', 'a', 'a']
        # top_a2 = ['b', 'b', 'b']  # tp = 2, fp = 0
        # top_p2 = ['b', 'b', 'a']
        t = {'tp': 2, 'fp': 1, 'tn': 0, 'fn': 0}
        b = {'tp': 2, 'fp': 0, 'tn': 0, 'fn': 1}

        # Calculate balanced accuracy
        top_sensitivity = t['tp'] / (t['tp'] + t['fn'])  # in the top, so positive is class a: tp / (tp + fn)
        top_specificity = t['tn'] / (t['tn'] + t['fp'])  # 0 / (0 + 1), tn / (tn + fp)
        top_bal_accuracy = (top_sensitivity + top_specificity) / 2

        bottom_sensitivity = b['tp'] / (b['tp'] + b['fn'])  # in the bottom, so positive is class b: tp / (tp + fn)
        bottom_specificity = 0  # 0 / (0 + 0), tn / (tn + fp)
        bottom_bal_accuracy = (bottom_sensitivity + bottom_specificity) / 2

        # Calculate the metric
        log_top_metric = math.log(top_bal_accuracy + 1)
        log_bottom_metric = math.log(bottom_bal_accuracy + 1)
        imbalance_penalty = math.log(abs(t['tp'] - b['tp']) + 1)
        metric_value = log_top_metric + log_bottom_metric - imbalance_penalty

        top_bottom_percent = .4
        # create our object
        m = TopAndBottomMetric(top_bottom_percent)
        self.assertTrue(metric_value == m._calc_balanced_accuracy_metric(t, b))

    def test_tp_fp_metric(self):
        # Test Data
        # answers = np.array(['b', 'b', 'b', 'b', 'a', 'b', 'a'])
        # predictions = np.array(['a', 'a', 'b', 'b', 'a', 'a', 'a'])
        # probabilities_list = [[.9, .1], [.4, .6], [.2, .8], [.3, .7], [.8, .2], [.55, .45], [.95, .05]]
        # probabilities = np.array([np.array(i) for i in probabilities_list])

        # Prepare the test data, create our top and bottom most probable confusion
        # matrices by hand from the above data
        top_a1 = ['a', 'b', 'a']  # tp = 2, fp = 1
        # top_p1 = ['a', 'a', 'a']
        top_a2 = ['b', 'b', 'b']  # tp = 2, fp = 0
        # top_p2 = ['b', 'b', 'a']
        t = {'tp': 2, 'fp': 1, 'tn': 0, 'fn': 0}
        b = {'tp': 2, 'fp': 0, 'tn': 0, 'fn': 1}

        # Compute the metric
        total_samples = len(top_a2) + len(top_a1)
        metric_value = ((t['tp'] - t['fp']) + (b['tp'] - b['fp']) - abs(t['tp'] - b['tp'])) / total_samples

        # Have the class compute the metric, then compare results
        top_bottom_percent = .4
        m = TopAndBottomMetric(top_bottom_percent)

        self.assertTrue(metric_value == m._calc_tp_fp_metric(t, b))

    def test_predictive_value_metric(self):
        # Test Data
        # answers = np.array(['b', 'b', 'b', 'b', 'a', 'b', 'a'])
        # predictions = np.array(['a', 'a', 'b', 'b', 'a', 'a', 'a'])
        # probabilities_list = [[.9, .1], [.4, .6], [.2, .8], [.3, .7], [.8, .2], [.55, .45], [.95, .05]]
        # probabilities = np.array([np.array(i) for i in probabilities_list])

        # Prepare the test data, create our top and bottom most probable confusion
        # matrices by hand from the above data
        # top_a1 = ['a', 'b', 'a']  # tp = 2, fp = 1
        # top_p1 = ['a', 'a', 'a']
        # top_a2 = ['b', 'b', 'b']  # tn = 2, fn = 0
        # top_p2 = ['b', 'b', 'a']
        cm = {'tp': 2, 'fn': 0, 'fp': 2, 'tn': 2}

        # Compute the metric
        ppv = cm['tp'] / (cm['tp'] + cm['fp'])
        npv = cm['tn'] / (cm['tn'] + cm['fn'])
        # print(f"{ppv} {npv}")
        metric_value = math.log(ppv + 1) + math.log(npv + 1) - abs((ppv - npv)/2)

        # Have the class compute the metric, then compare results
        top_bottom_percent = .4
        m = TopAndBottomMetric(top_bottom_percent)

        # print(m._calc_predictive_value_metric(cm))
        self.assertTrue(metric_value == m._calc_predictive_value_metric(cm))

    #
    # Test Estimator - this tests the full code path
    #

    def test_estimator_balanced_accuracy(self):
        data = np.array([1, 2, 1, 1, 2, 1, 2])
        answers = np.array(['b', 'b', 'b', 'b', 'b', 'b', 'a'])
        estimator = DummyClassifier(strategy="most_frequent")
        estimator.fit(data, answers)

        top_bottom_percent = .4
        m = TopAndBottomMetric(top_bottom_percent)

        # print(m.balanced_accuracy_metric(estimator, data, answers))
        self.assertTrue(np.isclose(m.balanced_accuracy_metric(estimator, data, answers), -0.28768))

    def test_estimator_tp_fp(self):
        data = np.array([1, 2, 1, 1, 2, 1, 2])
        answers = np.array(['b', 'b', 'b', 'b', 'b', 'b', 'a'])
        estimator = DummyClassifier(strategy="most_frequent")
        estimator.fit(data, answers)

        top_bottom_percent = .4
        m = TopAndBottomMetric(top_bottom_percent)

        # print(m.tp_fp_metric(estimator, data, answers))
        self.assertTrue(np.isclose(m.tp_fp_metric(estimator, data, answers), -0.166667))

    def test_estimator_precision(self):
        data = np.array([1, 2, 1, 1, 2, 1, 2])
        answers = np.array(['b', 'b', 'b', 'b', 'b', 'b', 'a'])
        estimator = DummyClassifier(strategy="most_frequent")
        estimator.fit(data, answers)

        top_bottom_percent = .4
        m = TopAndBottomMetric(top_bottom_percent)

        # print(m.precision_metric(estimator, data, answers))
        self.assertTrue(np.isclose(m.precision_metric(estimator, data, answers), -.587787))

    def test_estimator_predictive_value(self):
        data = np.array([1, 2, 1, 1, 2, 1, 2])
        answers = np.array(['b', 'b', 'b', 'b', 'b', 'b', 'a'])
        estimator = DummyClassifier(strategy="most_frequent")
        estimator.fit(data, answers)

        top_bottom_percent = .4
        m = TopAndBottomMetric(top_bottom_percent)

        # print(m.predictive_value_metric(estimator, data, answers))
        self.assertTrue(np.isclose(m.predictive_value_metric(estimator, data, answers), 0.189469))


if __name__ == '__main__':
    unittest.main()
