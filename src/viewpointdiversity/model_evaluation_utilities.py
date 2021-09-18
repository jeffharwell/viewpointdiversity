from operator import itemgetter

import numpy
from sklearn import metrics


def print_top_n_stats(answers, predictions, avg_rate_in_data, positive_class_label):
    """
    Prints the Stats for a given class label

    :param answers: an array of answers
    :param predictions: an array of predictions
    :param avg_rate_in_data: this is the percent of the total dataset that has this class values, used to calculate lift
    :param positive_class_label: the value of the label corresponding to the class we are printing stats for.
                                 This controls what class we consider positive, and put in the upper left quadrant
                                 of the confusion matrix
    """
    print("Printing top stats, positive class label = ", positive_class_label)
    # Calculate our true positives, false negatives, true negatives, and false positives
    tp = sum([1 if answers[i] == predictions[i] else 0 for i in range(0, len(predictions)) if
              predictions[i] == positive_class_label])
    fp = sum([1 if answers[i] != predictions[i] else 0 for i in range(0, len(predictions)) if
              predictions[i] == positive_class_label])
    fn = sum([1 if answers[i] != predictions[i] else 0 for i in range(0, len(predictions)) if
              predictions[i] != positive_class_label])
    tn = sum([1 if answers[i] == predictions[i] else 0 for i in range(0, len(predictions)) if
              predictions[i] != positive_class_label])
    print("True Positive: ", tp)
    print("False Positive: ", fp)
    print("False Negative: ", fn)
    print("True Negative: ", tn)

    # The Confusion Matrix
    actual_positive = numpy.array([tp, fn])
    actual_negative = numpy.array([fp, tn])
    print("\nConfusion Matrix:")
    print(numpy.vstack((actual_positive, actual_negative)))

    # Sensitivity, specificity, and balanced accuracy
    if (tp + fn) == 0:
        print("\nSensitivity: Undefined")
    else:
        sensitivity = tp / (tp + fn)  # how many positive results did it find out of all the positive results available
        print("\nSensitivity: ", sensitivity)

    if (tn + fp) == 0:
        print("Specificity: Undefined")
    else:
        specificity = tn / (
                tn + fp)  # how many negative results did it find out of all the negative results available
        print("Specificity: ", specificity)

    print("Balanced Accuracy:", metrics.balanced_accuracy_score(answers, predictions))

    # lift
    lift = (sum([1 for a in answers if a == positive_class_label]) / len(answers)) / avg_rate_in_data
    print("Lift: %.2f" % lift)


def print_combined_stats(answers, predictions, avg_rate_in_data_by_class, positive_class_label):
    print("Printing combined stats, positive class label = ", positive_class_label)
    # Calculate our true positives, false negatives, true negatives, and false positives
    tp = sum([1 if answers[i] == predictions[i] else 0 for i in range(0, len(predictions)) if
              predictions[i] == positive_class_label])
    fp = sum([1 if answers[i] != predictions[i] else 0 for i in range(0, len(predictions)) if
              predictions[i] == positive_class_label])
    fn = sum([1 if answers[i] != predictions[i] else 0 for i in range(0, len(predictions)) if
              predictions[i] != positive_class_label])
    tn = sum([1 if answers[i] == predictions[i] else 0 for i in range(0, len(predictions)) if
              predictions[i] != positive_class_label])
    print("True Positive: ", tp)
    print("False Positive: ", fp)
    print("False Negative: ", fn)
    print("True Negative: ", tn)

    # The Confusion Matrix
    actual_positive = numpy.array([tp, fn])
    actual_negative = numpy.array([fp, tn])
    print("\nConfusion Matrix:")
    print(numpy.vstack((actual_positive, actual_negative)))

    # Sensitivity, specificity, and balanced accuracy
    if (tp + fn) == 0:
        print("\nSensitivity: Undefined")
    else:
        sensitivity = tp / (tp + fn)  # how many positive results did it find out of all the positive results available
        print("\nSensitivity: ", sensitivity)

    if (tn + fp) == 0:
        print("Specificity: Undefined")
    else:
        specificity = tn / (
                tn + fp)  # how many negative results did it find out of all the negative results available
        print("Specificity: ", specificity)

    print("Balanced Accuracy:", metrics.balanced_accuracy_score(answers, predictions))

    # lift
    for class_label, avg_rate_in_data in avg_rate_in_data_by_class.items():
        lift = (sum([1 for a in answers if a == class_label]) / len(answers)) / avg_rate_in_data
        print("Lift for class %s: %.2f" % (class_label, lift))


def analyze_top_predictions(answers, predictions, probabilities, top_number):
    """
    Analyze the top predictions of the model.

    :param answers: numpy array of answers
    :param predictions: numpy array of predictions
    :param probabilities: numpy array of arrays, each element in the array being a class proability
    :param top_number: the number of top performers to analyze
    """
    if top_number > len(answers):
        raise RuntimeError("Specified number of data points to analyze %s is greater then the number of samples %s" % (
            top_number, len(answers)))

    # Unpack our probabilities
    prob1 = [p[0] for p in probabilities]
    prob2 = [p[1] for p in probabilities]

    # Create a new data structure so that we can sort
    all_data = list(zip(answers, predictions, prob1, prob2))

    # Sort by probability one, this will correspond to our first class, stance 0, oppose
    sorted_data_zeros_first = sorted(all_data, key=itemgetter(2),
                                     reverse=True)  # Highest prob of getting a 0 is sorted first

    # Sort by probability two, this will correspond to our second class, stance 1, support
    sorted_data_ones_first = sorted(all_data, key=itemgetter(3),
                                    reverse=True)  # Highest prob of getting a 1 is at the beginning

    # Class 0 - Slice of top performers
    answers_class_0 = [a[0] for a in sorted_data_zeros_first[0:top_number]]
    predictions_class_0 = [a[1] for a in sorted_data_zeros_first[0:top_number]]

    # Class 1 - Slice of top performers
    answers_class_1 = [a[0] for a in sorted_data_ones_first[0:top_number]]
    predictions_class_1 = [a[1] for a in sorted_data_ones_first[0:top_number]]

    # Calculate our data data set statistics
    total_count_class_0 = sum([1 for a in answers if a == 0])
    total_count_class_1 = sum([1 for a in answers if a == 1])
    total_percent_class_0 = total_count_class_0 / len(answers)
    total_percent_class_1 = total_count_class_1 / len(answers)

    print("#\n# For Class 0 the top (most probably positive) %s datapoints:\n#\n" % top_number)
    print_top_n_stats(answers_class_0, predictions_class_0, total_percent_class_0, 0)
    print("\n#\n# For Class 1 the bottom (most probably negative) %s datapoints:\n#\n" % top_number)
    print_top_n_stats(answers_class_1, predictions_class_1, total_percent_class_1, 0)
    print("\n#\n# Combined Confusion Matrix:")
    print("# The %s data points predicted most likely to be class 0" % top_number)
    print("# and the %s datapoints predicted most likely to be class 1." % top_number)
    print("#\n")
    print_combined_stats(answers_class_0 + answers_class_1, predictions_class_0 + predictions_class_1,
                         {0: total_percent_class_0, 1: total_percent_class_1}, 0)


# Analyze the top predictions of the model.
#
# answers = numpy array of answers
# predictions = numpy array of predictions
# probablities = numpy array of arrays, each element in the array being a class proability
# cut_off = analyze all results with probability greater than the cut_off value
def analyze_top_percentage(answers, predictions, probabilities, cut_off):
    if cut_off > 1 or cut_off <= 0:
        raise RuntimeError("Cutoff percentage must be a decimal between 0 and 1")

    # Unpack our probabilities
    prob1 = [p[0] for p in probabilities]
    prob2 = [p[1] for p in probabilities]

    # Create a new data structure so that we can sort
    all_data = list(zip(answers, predictions, prob1, prob2))

    # Sort by probability one, this will correspond to our first class, stance 0, oppose
    sorted_data_zeros_first = sorted(all_data, key=itemgetter(2),
                                     reverse=True)  # Highest prob of getting a 0 is sorted first

    # Sort by probability two, this will correspond to our second class, stance 1, support
    sorted_data_ones_first = sorted(all_data, key=itemgetter(3),
                                    reverse=True)  # Highest prob of getting a 1 is at the beginning

    # Class 0 - Slice of top performers
    answers_class_0 = [a[0] for a in sorted_data_zeros_first if a[2] > cut_off]
    predictions_class_0 = [a[1] for a in sorted_data_zeros_first if a[2] > cut_off]

    # Class 1 - Slice of top performers
    answers_class_1 = [a[0] for a in sorted_data_ones_first if a[3] > cut_off]
    predictions_class_1 = [a[1] for a in sorted_data_ones_first if a[3] > cut_off]

    # Calculate our data data set statistics
    total_count_class_0 = sum([1 for a in answers if a == 0])
    total_count_class_1 = sum([1 for a in answers if a == 1])
    total_percent_class_0 = total_count_class_0 / len(answers)
    total_percent_class_1 = total_count_class_1 / len(answers)

    print("\nAnalyzing Probabilities > %s" % cut_off)
    print("Cutoff at %s results in:" % cut_off)
    print("     %s data points for class 0" % len(answers_class_0))
    print("     %s data points for class 1.\n" % len(answers_class_1))
    print("#\n# For Class 0:\n#\n")
    print_top_n_stats(answers_class_0, predictions_class_0, total_percent_class_0, 0)
    print("\n\n#\n# For Class 1:\n#\n")
    print_top_n_stats(answers_class_1, predictions_class_1, total_percent_class_1, 1)
    print("\n#\n# Combined Confusion Matrix:")
    print("# The %s data points predicted most likely to be class 0" % len(answers_class_0))
    print("# and the %s datapoints predicted most likely to be class 1." % len(answers_class_1))
    print("#\n")
    print_combined_stats(answers_class_0 + answers_class_1, predictions_class_0 + predictions_class_1,
                         {0: total_percent_class_0, 1: total_percent_class_1}, 0)


def custom_metric(estimator, data, answers):
    """
    Grid Search Custom Metric

    Custom metric for use with a grid search, it rewards the ability to detect the minority class (True Negatives)
    in the bottom 10% of the predictions when they are sorted by descending probability that they are a majority class
    member. Note that the classifier does need to be able to output probabilities in order for this metric to work.

    Example of Use for an SVM Classifier

        grid = GridSearchCV(svm.SVC(probability=True),
                            param_grid={'C': [.00001, .001, 1, 10, 100],
                            'gamma':[1e-4, 1e-2, 1, 10]},
                            scoring=custom_metric,cv=5)
        grid.fit(train_x, train_y)
        print("Best Params: ",grid.best_params_)
        best_c = grid.best_params_['C']
        best_gamma = grid.best_params_['gamma']

    :param estimator: A scikitlearn estimator
    :param data: Data that we are classifying
    :param answers: Correct answers for the data we are classifying
    """

    def calculate_confusion_matrix(ans, pred, class_label):
        # print(pred)
        tp = sum([1 if ans[i] == pred[i] else 0 for i in range(0, len(pred)) if pred[i] == class_label])
        fp = sum([1 if ans[i] != pred[i] else 0 for i in range(0, len(pred)) if pred[i] == class_label])
        fn = sum([1 if ans[i] != pred[i] else 0 for i in range(0, len(pred)) if pred[i] != class_label])
        tn = sum([1 if ans[i] == pred[i] else 0 for i in range(0, len(pred)) if pred[i] != class_label])
        return tp, fp, tn, fn

    predictions = estimator.predict(data)
    probabilities = estimator.predict_proba(data)
    top_number = round(.1 * len(data))

    # Unpack our probabilities
    prob1 = [p[0] for p in probabilities]
    prob2 = [p[1] for p in probabilities]

    # Create a new data structure so that we can sort
    all_data = list(zip(answers, predictions, prob1, prob2))

    # Sort by probability one, this will correspond to our first class, stance 0, oppose
    sorted_data_zeros_first = sorted(all_data, key=itemgetter(2),
                                     reverse=True)  # Highest prob of getting a 0 is sorted first

    # Sort by probability two, this will correspond to our second class, stance 1, support
    sorted_data_ones_first = sorted(all_data, key=itemgetter(3),
                                    reverse=True)  # Highest prob of getting a 1 is at the beginning

    # Class 0 - Slice of top performers
    answers_class_0 = [a[0] for a in sorted_data_zeros_first[0:top_number]]
    predictions_class_0 = [a[1] for a in sorted_data_zeros_first[0:top_number]]

    # Class 1 - Slice of top performers
    answers_class_1 = [a[0] for a in sorted_data_ones_first[0:top_number]]
    predictions_class_1 = [a[1] for a in sorted_data_ones_first[0:top_number]]

    # Calculate our true positives, false negatives, true negatives, and false positives
    # (tp_top, fp_top, tn_top, fn_top) = calculate_confusion_matrix(answers_class_0, predictions_class_0, 0)
    (tp_bottom, fp_bottom, tn_bottom, fn_bottom) = calculate_confusion_matrix(answers_class_1, predictions_class_1, 0)

    if tn_bottom + fp_bottom == 0:
        if tn_bottom == 0:
            bottom_sensitivity = 0
        else:
            bottom_sensitivity = .9
    else:
        bottom_sensitivity = tn_bottom / (tn_bottom + fp_bottom)

    top_balanced_accuracy = metrics.balanced_accuracy_score(answers_class_0, predictions_class_0)
    bottom_balanced_accuracy = metrics.balanced_accuracy_score(answers_class_1, predictions_class_1)
    overall_balanced_accuracy = (top_balanced_accuracy + bottom_balanced_accuracy) / 2

    # We add a weight to balanced sensitivity. Overall accuracy is good, but sensitivity is more important in this case.
    overall_metric = 2 * bottom_sensitivity + overall_balanced_accuracy
    return overall_metric
