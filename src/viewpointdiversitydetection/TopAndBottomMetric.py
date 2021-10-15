import numpy as np
from operator import itemgetter
import math


class TopAndBottomMetric:
    """
    Returns a metric that can be used by a GridSearch to optimize the Top and Bottom Precision of a classifier.
    """

    def __init__(self, top_and_bottom_percent):
        """
        Initialize the class. This constructor returns a classifier.

        :param top_and_bottom_percent: the precentage of the data to consider when taking the top and bottom results
        """

        if top_and_bottom_percent <= 0 or top_and_bottom_percent >= .5:
            raise ValueError("Cannot have a top_and_bottom_percent that is less than 0 or .5 or greater.")
        self.top_and_bottom_percent = top_and_bottom_percent

    @staticmethod
    def _calculate_confusion_matrix(ans, pred, class_label):
        """
        Calculate a confusion matrix with 'class_label' as the positive class

        :param ans: list of correct class labels for each data point
        :param pred: list of predicted class labels for each data point
        :param class_label: the label for the positive class
        """

        tp = sum([1 if ans[i] == pred[i] else 0 for i in range(0, len(pred)) if pred[i] == class_label])
        fp = sum([1 if ans[i] != pred[i] else 0 for i in range(0, len(pred)) if pred[i] == class_label])
        fn = sum([1 if ans[i] != pred[i] else 0 for i in range(0, len(pred)) if pred[i] != class_label])
        tn = sum([1 if ans[i] == pred[i] else 0 for i in range(0, len(pred)) if pred[i] != class_label])

        return tp, fp, tn, fn

    @staticmethod
    def _get_results_from_estimator(estimator, data):
        """
        Uses the estimator that has been passed to us to generate both predictions and probabilities

        :param estimator: a Scikitlearn estimator
        :param data: data to pass to the estimator
        :return: tuple (predictions, probabilities)
        """
        predictions = estimator.predict(data)
        # documentation "Returns the probability of the samples for each class in the model. The columns
        #                correspond to the classes in sorted order, as they appear in the attribute classes_"
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.predict_proba
        probabilities = estimator.predict_proba(data)
        return predictions, probabilities

    @staticmethod
    def _get_classes_from_estimator(estimator):
        """
        Get our two class labels from the estimator. The order corresponds to how the probabilities
        are ordered as return by the get_results_from_estimator method.

        :param estimator: A scikitlearn estimator
        :return: tuple with two class labels (class_label_1, class_label_2)
        """
        class_labels = estimator.classes_

        if len(class_labels) != 2:
            raise RuntimeError("This metric only works for binary classification")
        class_label_1 = class_labels[0]
        class_label_2 = class_labels[1]

        return class_label_1, class_label_2

    @staticmethod
    def _get_top_n_most_probable(top_number, answers, predictions, prediction_probabilities):
        """
        Return a tuple with the top_n most certain answers and predictions as sorted by
        probability from highest to lowest. Note the the prediction_probabilities array should
        be the probabilities of the class you are interested in. So if you want the top n predictions
        for class 1 the prediction_probabilities array should contain the class 1 probabilities.

        :param top_number: The number of top results to return
        :param answers: An array of answers
        :param predictions: An array of predictions
        :prediction_probabilities: An array of probabilities for the predictions for the class we are interested in
        :return: tuple of n length arrays that have the most probable predictons and corresponding
                 answers (answers, predictions)
        """

        if len(answers) != len(predictions) != len(prediction_probabilities):
            raise RuntimeError("You must pass an equal number of answers, predictions, and probabilities.")

        if top_number > len(answers):
            raise RuntimeError(f"Your top_n value {top_number} must be less than the number of data points")

        # Sort all of the answers and predictions by probability descending and grab the first top_n
        zipped_data = list(zip(answers, predictions, prediction_probabilities))
        sorted_data = sorted(zipped_data, key=itemgetter(2), reverse=True)
        top_n_answers = [a[0] for a in sorted_data[0:top_number]]
        top_n_predictions = [a[1] for a in sorted_data[0:top_number]]

        return top_n_answers, top_n_predictions

    def _get_top_bottom_predictions_from_estimator(self, n, estimator, data, answers):
        """
        Run the estimator against the data and return the top and bottom n most certain
        predictions from the estimator.

        :param n: the number of data points to consider for the confusion matrix
        :param estimator: a scikitlearn estimator object
        :param data: data the estimator will use to make predictions
        :param answers: results dictionary for both classes of the structure
                {'class_1': {'answers':answers_class_1, 'predictions':predictions_class_1, 'label':class_label_1} ..}
        """
        top_number = n
        if top_number < 1:
            msg = f"Cannot create a confusion matrix using less than 1 data point"
            raise ValueError(msg)

        # Use the estimator to get our predictions, probabilities, and classes
        predictions, probabilities = self._get_results_from_estimator(estimator, data)
        class_label_1, class_label_2 = self._get_classes_from_estimator(estimator)

        # Unpack our probabilities
        # class 1 matches probability 1, class 2 goes with probability 2
        prob_1 = [p[0] for p in probabilities]
        prob_2 = [p[1] for p in probabilities]

        # for some classifiers (esp. the DummyClassifiers) the probabilities will be identical for each
        # sample, which means that the sort will not change the order and you end up looking at the same
        # n samples for the top and bottom. So reverse the answers and predictions for the class 2 most
        # probably so at least we look at a different number of samples.
        reversed_answers = np.flip(answers)
        reversed_predictions = np.flip(predictions)

        # Get most probably answers and predictions for both classes
        answers_class_1, predictions_class_1 = self._get_top_n_most_probable(top_number, answers, predictions, prob_1)
        answers_class_2, predictions_class_2 = self._get_top_n_most_probable(top_number, reversed_answers,
                                                                             reversed_predictions, prob_2)
        results = {'class_1': {'answers': answers_class_1, 'predictions': predictions_class_1, 'label': class_label_1},
                   'class_2': {'answers': answers_class_2, 'predictions': predictions_class_2, 'label': class_label_2}}

        return results

    def _get_top_bottom_confusion_matrix_from_estimator(self, n, estimator, data, answers):
        """
        Get the Top and Bottom Confusion Matrix from an Estimator

        Use the estimator to create predictions for the data. Then rank those predictions by
        probability and take the most probable n predictions and return a confusion matrix
        representing the classifier performance when it is most certain.

        :param n: the number of data points to consider for the confusion matrix
        :param estimator: a scikitlearn estimator object
        :param data: data the estimator will use to make predictions
        :param answers: the correct classification for each row of data
        :return: a tuple of dictionaries, the first of which is the confusion matrix for the top n
                 most certain predictions for class 1, the second is the same but for class 2.
        """
        r = self._get_top_bottom_predictions_from_estimator(n, estimator, data, answers)

        # Calculate our true positives, false negatives, true negatives, and false positives
        (tp_top, fp_top, tn_top, fn_top) = self._calculate_confusion_matrix(r['class_1']['answers'],
                                                                            r['class_1']['predictions'],
                                                                            r['class_1']['label'])
        (tp_bottom, fp_bottom, tn_bottom, fn_bottom) = self._calculate_confusion_matrix(r['class_2']['answers'],
                                                                                        r['class_2']['predictions'],
                                                                                        r['class_2']['label'])
        top_cm = {'tp': tp_top, 'fp': fp_top, 'tn': tn_top, 'fn': fn_top}
        bottom_cm = {'tp': tp_bottom, 'fp': fp_bottom, 'tn': tn_bottom, 'fn': fn_bottom}

        return top_cm, bottom_cm

    def _get_top_bottom_combined_confusion_matrix_from_estimator(self, n, estimator, data, answers):
        """
        Get the combined Top and Bottom Confusion Matrix from an Estimator

        Use the estimator to create predictions for the data. Then rank those predictions by
        probability and take the most probable n predictions and return a confusion matrix
        representing the classifier performance when it is most certain.

        :param n: the number of data points to consider for the confusion matrix
        :param estimator: a scikitlearn estimator object
        :param data: data the estimator will use to make predictions
        :param answers: the correct classification for each row of data
        :return: a tuple of dictionaries, the first of which is the confusion matrix for the top n
                 most certain predictions for class 1, the second is the same but for class 2.
        """
        r = self._get_top_bottom_predictions_from_estimator(n, estimator, data, answers)

        # Calculate our true positives, false negatives, true negatives, and false positives
        answers = r['class_1']['answers'] + r['class_2']['answers']
        predictions = r['class_1']['predictions'] + r['class_2']['predictions']
        (tp, fp, tn, fn) = self._calculate_confusion_matrix(answers, predictions, r['class_1']['label'])
        cm = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

        return cm

    @staticmethod
    def _calc_sensitivity(tp, fn):
        """
        Calculates sensitivity. Returns 0 if tp + fn == 0. This is also known as the True Positive Rate (TPR)

        :param tp: true positive count
        :param fn: false negative count
        :returns: tp / (tp + fn) or 0 if tn + fp == 0
        """
        if tp + fn == 0:
            sensitivity = 0
        else:
            sensitivity = tp / (tp + fn)

        return sensitivity

    @staticmethod
    def _calc_specificity(tn, fp):
        """
        Calculates specificity. Returns 0 if tn + fp == 0. This is also known as the True Negative Rate (TNR)

        :param tn: true negative count
        :param fp: false positive count
        :returns: tn / (tn + fp) or 0 if tn + fp == 0
        """
        if tn + fp == 0:
            specificity = 0
        else:
            specificity = tn / (tn + fp)

        return specificity

    @staticmethod
    def _calc_precision(tp, fp):
        """
        Calculates precision. Returns 0 if tp + fp == 0.

        :param tp: true positive count
        :param fp: false positive count
        :returns: tp / (tp + fp) or 0 if tp + fp == 0
        """
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)

        return precision

    @staticmethod
    def _calc_predictive_value(true_counts, false_counts):
        """
        Calculates the predictive value. If you pass it positive true and false
        counts you get the positive predictive value (PPV which is the same as
        precision), if you pass it the negative true and false counts you get
        the negative predictive value (NPV).

        :param true_counts: the number of true classifications, can be either negative or positive
        :param false_counts: the number of false/incorrect classification, either negative or positive
        :return: the predictive value true / (true + false), or 0 if true + false == 0
        """
        if true_counts + false_counts == 0:
            pv = 0
        else:
            pv = true_counts / (true_counts + false_counts)

        return pv

    """
    A note on class design here. There are two major pieces of the metric algorithms. The first involves
    using the estimator, data, and answers to get the predictions from the estimator, sort them by probability
    by class, and create a confusion matrix from the top n results from each class. The second part of the 
    algorithm involves using those confusion matrices to calculate a performance metric.

    The first part of the alogorithm is largely procedural and involves manipulating a Scikit learn Estimator
    object and its output. All of these maniplulations are in the primary method call: "precision_metric",
    "tp_fp_metric", and "balanced_accuracy_metric". I don't have test harnesses for this part of the algorithm
    due to the law of diminishing returns; the algorithm works if the estimator behaves as expected.

    The second part of the algorithm is a calculation that we can verify using a test harness. This is why it
    is broken out into its own method: "_calc_precisios_metric", "_calc_balanced_accuracy_metric", and 
    "calc_tp_fp_metric". This architecture allows us to test the calculation portion of the metric independent
    of the classifier.
    """

    def _calc_precision_metric(self, top_confusion_matrix, bottom_confusion_matrix):
        """
        Calculates the precision metrix from the top and bottom confusion matrix

        :param top_confusion_matrix: dictionary with tp, fp, tn, fn keys and values
        :param bottom_confusion_matrix: dictionary with tp, fp, tn, fn keys and values
        :returns: ln(top_precision + 1) + ln(bottom_precision + 1) - ln(|top_true_positive - bottom_true_positive| + 1)
        """
        t = top_confusion_matrix
        b = bottom_confusion_matrix

        top_precision = self._calc_precision(t['tp'], t['fp'])
        bottom_precision = self._calc_precision(b['tp'], b['fp'])

        log_top_metric = math.log(top_precision + 1)
        log_bottom_metric = math.log(bottom_precision + 1)
        imbalance_penalty = math.log(abs(t['tp'] - b['tp']) + 1)

        metric_value = log_top_metric + log_bottom_metric - imbalance_penalty

        return metric_value

    def precision_metric(self, estimator, data, answers):
        """
        Calculate the top and bottom precision of the given predictions order by probability.

        Example of Use for an SVM Classifier

            metric_obj = TopAndBottomMetric(.1)
            grid = GridSearchCV(svm.SVC(probability=True),
                                param_grid={'C': [.00001, .001, 1, 10, 100],
                                'gamma':[1e-4, 1e-2, 1, 10]},
                                scoring=metric_obj.precision_metric, cv=5)
            grid.fit(train_x, train_y)
            print("Best Params: ",grid.best_params_)
            best_c = grid.best_params_['C']
            best_gamma = grid.best_params_['gamma']

        :param estimator: a scikitlearn estimator
        :param data: data to be processed by the estimator
        :param answers: an array of correct classifications for the data
        :returns: ln(top_precision + 1) + ln(bottom_precision + 1) - ln(|top_true_positive - bottom_true_positive| + 1)
        """

        top_number = round(self.top_and_bottom_percent * len(answers))  # 10% of the data
        if top_number < 1:
            msg = f"Top and bottom percent of {self.top_and_bottom_percent} resulted in selecting less that one sample."
            raise ValueError(msg)

        t, b = self._get_top_bottom_confusion_matrix_from_estimator(top_number, estimator, data, answers)

        return self._calc_precision_metric(t, b)

    @staticmethod
    def _calc_tp_fp_metric(top_confusion_matrix, bottom_confusion_matrix):
        """
        Calculates the true positive, false positive metric from the top and bottom confusion
        matrix.

        :param top_confusion_matrix: confusion matrix from the top most probable class 1 predictions,
                                     class 1 is positive.
        :param bottom_confusion_matrix: confusion matrix from the bottom most probable class 2 predictions,
                                        class 2 is positive.
        """
        t = top_confusion_matrix
        b = bottom_confusion_matrix
        total_t_samples = sum([s for s in t.values()])
        total_b_samples = sum([s for s in b.values()])
        total_samples = total_t_samples + total_b_samples

        metric_value = ((t['tp'] - t['fp']) + (b['tp'] - b['fp']) - abs(t['tp'] - b['tp'])) / total_samples

        return metric_value

    def tp_fp_metric(self, estimator, data, answers):
        """
        Calculate a custom metric based on true positives and false positives.
        Because we are only considering the top n most probable predictions from both classes there will
        probably not be any negative guesses, so tn and fn shouldn't come into play, we are really
        trying to optimize getting equal numbers of true predictions from both classes while minimizing
        the number of false positivies.

        Example of Use for an SVM Classifier

            metric_obj = TopAndBottomMetric(.1)
            grid = GridSearchCV(svm.SVC(probability=True),
                                param_grid={'C': [.00001, .001, 1, 10, 100],
                                'gamma':[1e-4, 1e-2, 1, 10]},
                                scoring=metric_obj.tp_fp_metric, cv=5)
            grid.fit(train_x, train_y)
            print("Best Params: ",grid.best_params_)
            best_c = grid.best_params_['C']
            best_gamma = grid.best_params_['gamma']

        :param estimator: a scikitlearn estimator
        :param data: data to be processed by the estimator
        :param answers: an array of correct classifications for the data
        :return: (top_tp - top_fp) + (bottom_tp - bottom_fp) - |top_tp - bottom_tp|
        """
        top_number = round(self.top_and_bottom_percent * len(answers))  # 10% of the data
        if top_number < 1:
            msg = f"Top and bottom percent of {self.top_and_bottom_percent} resulted in selecting less that one sample."
            raise ValueError(msg)

        # t is a dictionary with the top confusion matrix, b is a dictionary containing the bottom
        # confusion matrix entries
        t, b = self._get_top_bottom_confusion_matrix_from_estimator(top_number, estimator, data, answers)

        return self._calc_tp_fp_metric(t, b)

    def _calc_balanced_accuracy_metric(self, top_confusion_matrix, bottom_confusion_matrix):
        """
        Calculate the top and bottom balanced accuracy metric from the top and bottom confusion
        matrix.

        :param top_confusion_matrix: confusion matrix from the top most probable class 1 predictions,
                                     class 1 is positive.
        :param bottom_confusion_matrix: confusion matrix from the bottom most probable class 2 predictions,
                                        class 2 is positive.
        """
        t = top_confusion_matrix
        b = bottom_confusion_matrix

        t_sensitivity = self._calc_sensitivity(t['tp'], t['fn'])
        t_specificity = self._calc_specificity(t['tn'], t['fp'])
        t_balanced_accuracy = (t_sensitivity + t_specificity) / 2

        b_sensitivity = self._calc_sensitivity(b['tp'], b['fn'])
        b_specificity = self._calc_specificity(b['tn'], b['fp'])
        b_balanced_accuracy = (b_sensitivity + b_specificity) / 2

        log_top_metric = math.log(t_balanced_accuracy + 1)
        log_bottom_metric = math.log(b_balanced_accuracy + 1)
        imbalance_penalty = math.log(abs(t['tp'] - b['tp']) + 1)
        metric_value = log_top_metric + log_bottom_metric - imbalance_penalty

        return metric_value

    def balanced_accuracy_metric(self, estimator, data, answers):
        """
        Create a metric based on the top and bottom balanced accuracy and return it.

            Example of Use for an SVM Classifier

            metric_obj = TopAndBottomMetric(.1)
            grid = GridSearchCV(svm.SVC(probability=True),
                                param_grid={'C': [.00001, .001, 1, 10, 100],
                                'gamma':[1e-4, 1e-2, 1, 10]},
                                scoring=metric_obj.balanced_accuracy_metric, cv=5)
            grid.fit(train_x, train_y)
            print("Best Params: ",grid.best_params_)
            best_c = grid.best_params_['C']
            best_gamma = grid.best_params_['gamma']

        :param answers: an array of correct answers
        :param estimator: a scikitlearn estimator
        :param data: data to be processed by the estimator
        :param answers: an array of correct classifications for the data
        :return: ln(top_balanced_accuracy + 1) + ln(bottom_balanced_accuracy + 1) -
                 ln(|top_true_positive - bottom_true_positive| + 1)
        """

        top_number = round(self.top_and_bottom_percent * len(answers))  # 10% of the data
        if top_number < 1:
            msg = f"Top and bottom percent of {self.top_and_bottom_percent} resulted in selecting less that one sample."
            raise ValueError(msg)

        t, b = self._get_top_bottom_confusion_matrix_from_estimator(top_number, estimator, data, answers)

        return self._calc_balanced_accuracy_metric(t, b)

    def _calc_predictive_value_metric(self, confusion_matrix):
        """
        Calculate the top and bottom balanced accuracy metric from the top and bottom confusion
        matrix.

        :param confusion_matrix: combined confusion matrix of the form
                                 {'tp':tp, 'fp':fp, 'tn':tn, 'fn':fn}
        """
        ppv = self._calc_predictive_value(confusion_matrix['tp'], confusion_matrix['fp'])
        npv = self._calc_predictive_value(confusion_matrix['tn'], confusion_matrix['fn'])
        imbalance_penalty = abs((ppv - npv) / 2)
        metric_value = math.log(ppv + 1) + math.log(npv + 1) - imbalance_penalty

        return metric_value

    def predictive_value_metric(self, estimator, data, answers):
        """
        Create a metric based on the top and bottom balanced accuracy and return it.

            Example of Use for an SVM Classifier

            metric_obj = TopAndBottomMetric(.1)
            grid = GridSearchCV(svm.SVC(probability=True),
                                param_grid={'C': [.00001, .001, 1, 10, 100],
                                'gamma':[1e-4, 1e-2, 1, 10]},
                                scoring=metric_obj.balanced_accuracy_metric, cv=5)
            grid.fit(train_x, train_y)
            print("Best Params: ",grid.best_params_)
            best_c = grid.best_params_['C']
            best_gamma = grid.best_params_['gamma']

        :param estimator: a scikitlearn estimator
        :param data: data to be processed by the estimator
        :param answers: an array of correct classifications for the data
        :return: ln(top_balanced_accuracy + 1) + ln(bottom_balanced_accuracy + 1) -
                 ln(|top_true_positive - bottom_true_positive| + 1)
        """

        top_number = round(self.top_and_bottom_percent * len(answers))  # 10% of the data
        if top_number < 1:
            msg = f"Top and bottom percent of {self.top_and_bottom_percent} resulted in selecting less that one sample."
            raise ValueError(msg)
        cm = self._get_top_bottom_combined_confusion_matrix_from_estimator(top_number, estimator, data, answers)

        return self._calc_predictive_value_metric(cm)
    