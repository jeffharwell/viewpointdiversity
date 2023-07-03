from operator import itemgetter

import numpy
from sklearn import metrics
from sklearn.metrics import confusion_matrix


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
    print(f"Positive class label: '{positive_class_label}'")
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
        print(f"\nSensitivity/TPR: {sensitivity:.2f}")

    if (tn + fp) == 0:
        print("Specificity/TNR: Undefined")
    else:
        specificity = tn / (tn + fp)  # how many negative results did it find out of all the negative results available
        print(f"Specificity/TNR: {specificity:.2f}")

    if (tp + fp) == 0:
        print("Precision: Undefined")
    else:
        precision = tp / (tp + fp)
        print(f"Precision:       {precision:.2f}")

    print(f"Balanced Accuracy: {metrics.balanced_accuracy_score(answers, predictions):.2f}")

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
        print(f"\nSensitivity/TPR: {sensitivity:.2f}")

    if (tn + fp) == 0:
        print("Specificity: Undefined")
    else:
        specificity = tn / (tn + fp)  # how many negative results did it find out of all the negative results available
        print(f"Specificity/TNR: {specificity:.2f}")

    print("Balanced Accuracy:", metrics.balanced_accuracy_score(answers, predictions))

    # lift
    for class_label, avg_rate_in_data in avg_rate_in_data_by_class.items():
        lift = (sum([1 for a in answers if a == class_label]) / len(answers)) / avg_rate_in_data
        print(f"Lift for class {class_label}: {lift:.2f}")


def analyze_top_predictions(answers, predictions, probabilities, top_number, class_1_label, class_2_label):
    """
    Analyze the top predictions of the model.

    :param answers: numpy array of answers
    :param predictions: numpy array of predictions
    :param probabilities: numpy array of arrays, each element in the array being a class probability
    :param top_number: the number of top performers to analyze
    :param class_1_label: The label assigned to the second class
    :param class_2_label: The label assigned to the second class
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

    reversed_data = all_data.copy()
    reversed_data.reverse()
    # Sort by probability two, this will correspond to our second class, stance 1, support
    sorted_data_ones_first = sorted(reversed_data, key=itemgetter(3),
                                    reverse=True)  # Highest prob of getting a 1 is at the beginning

    # Class 0 - Slice of top performers
    answers_class_1 = [a[0] for a in sorted_data_zeros_first[0:top_number]]
    predictions_class_1 = [a[1] for a in sorted_data_zeros_first[0:top_number]]

    # Class 1 - Slice of top performers
    answers_class_2 = [a[0] for a in sorted_data_ones_first[0:top_number]]
    predictions_class_2 = [a[1] for a in sorted_data_ones_first[0:top_number]]

    # Calculate our data data set statistics
    total_count_class_1 = sum([1 for a in answers if a == class_1_label])
    total_count_class_2 = sum([1 for a in answers if a == class_2_label])
    total_percent_class_1 = total_count_class_1 / len(answers)
    total_percent_class_2 = total_count_class_2 / len(answers)

    print(f"#\n# For Class '{class_1_label}' the top (most probably positive) {top_number} datapoints:\n#\n")
    print_top_n_stats(answers_class_1, predictions_class_1, total_percent_class_1, class_1_label)
    print(f"\n#\n# For Class '{class_2_label}' the bottom (most probably negative) {top_number} datapoints:\n#\n")
    print_top_n_stats(answers_class_2, predictions_class_2, total_percent_class_2, class_2_label)
    print("\n#\n# Combined Confusion Matrix:")
    print(f"# The {top_number} data points predicted most likely to be class {class_1_label}")
    print(f"# and the {top_number} datapoints predicted most likely to be class {class_2_label}.")
    print("#\n")
    print_combined_stats(answers_class_1 + answers_class_2, predictions_class_1 + predictions_class_2,
                         {class_1_label: total_percent_class_1, class_2_label: total_percent_class_2}, class_1_label)


def calculate_stats(answers, predictions, cm):
    """
    Prints the Stats for a given class label

    :param answers: an array of answers
    :param predictions: an array of predictions
    :param cm: confusion matrix of form [[TP, FN], [FP, TN]]
    """
    # confusion matrix looks like this [[ 23, 63], [ 40, 260]]
    #                                  [[TP, FN], [FP, TN]]
    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]

    stats = {}

    # Sensitivity, specificity, and balanced accuracy
    if (tp + fn) == 0:
        tpr = 'Undefined'
    else:
        tpr_value = tp / (tp + fn)  # how many positive results did it find out of all the positive results available
        tpr = f"{tpr_value:.2f}"
    stats['TPR'] = tpr

    if (tn + fp) == 0:
        print("Specificity/TNR: Undefined")
        tnr = 'Undefined'
    else:
        tnr_value = tn / (tn + fp)  # how many negative results did it find out of all the negative results available
        tnr = f"{tnr_value:.2f}"
    stats['TNR'] = tnr

    if (tp + fp) == 0:
        ppv = 'Undefined'
    else:
        ppv_value = tp / (tp + fp)
        ppv = f"{ppv_value:.2f}"
    stats['PPV'] = ppv

    if (tn + fn) == 0:
        npv = 'Undefined'
    else:
        npv_value = tn / (tn + fn)
        npv = f"{npv_value:.2f}"
    stats['NPV'] = npv

    bal_acc = f"{metrics.balanced_accuracy_score(answers, predictions):.2f}"
    stats['Bal Acc'] = bal_acc

    return stats


def calculate_stats_as_float(answers, predictions, cm):
    """
    Prints the Stats for a given class label, but return floating point values, not strings.
    If a value is undefined return -1.

    :param answers: an array of answers
    :param predictions: an array of predictions
    :param cm: confusion matrix of form [[TP, FN], [FP, TN]]
    """
    # confusion matrix looks like this [[ 23, 63], [ 40, 260]]
    #                                  [[TP, FN], [FP, TN]]
    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]

    stats = {}

    # Sensitivity, specificity, and balanced accuracy
    if (tp + fn) == 0:
        tpr = -1
    else:
        tpr_value = tp / (tp + fn)  # how many positive results did it find out of all the positive results available
        tpr = tpr_value
    stats['TPR'] = tpr

    if (tn + fp) == 0:
        print("Specificity/TNR: Undefined")
        tnr = -1
    else:
        tnr_value = tn / (tn + fp)  # how many negative results did it find out of all the negative results available
        tnr = tnr_value
    stats['TNR'] = tnr

    if (tp + fp) == 0:
        ppv = -1
    else:
        ppv_value = tp / (tp + fp)
        ppv = ppv_value
    stats['PPV'] = ppv

    if (tn + fn) == 0:
        npv = -1
    else:
        npv_value = tn / (tn + fn)
        npv = npv_value
    stats['NPV'] = npv

    bal_acc = metrics.balanced_accuracy_score(answers, predictions)
    stats['Bal Acc'] = bal_acc

    return stats


def create_run_stats(answers, predictions, probabilities, top_number, class_1_label, class_2_label):
    """
    This function uses some functions from the vdd module to create a some statistics about the effectiveness
    of the model for the top and bottom most certain predictions. It is based off of the code from the
    generate_markdown_table function just extracted from some of the string generation logic.

    This allows us to get the basic top and bottom statistics as a data structure, rather than being embedded
    in a Markdown table string.

    :param answers: numpy array of answers
    :param predictions: numpy array of predictions
    :param probabilities: numpy array of arrays, each element in the array being a class probability
    :param top_number: the number of top performers to analyze
    :param class_1_label: the label assigned to the first class
    :param class_2_label: the label assigned to the second class
    :return stats: a dictionary of stats
    """
    # Unpack our probabilities
    prob1 = [p[0] for p in probabilities]
    prob2 = [p[1] for p in probabilities]

    # Create a new data structure so that we can sort
    all_data = list(zip(answers, predictions, prob1, prob2))

    # Sort by probability one, this will correspond to our first class, stance 0, oppose
    sorted_data_zeros_first = sorted(all_data, key=itemgetter(2),
                                     reverse=True)  # Highest prob of getting a 0 is sorted first

    reversed_data = all_data.copy()
    reversed_data.reverse()
    # Sort by probability two, this will correspond to our second class, stance 1, support
    sorted_data_ones_first = sorted(reversed_data, key=itemgetter(3),
                                    reverse=True)  # Highest prob of getting a 1 is at the beginning

    # Class 0 - Slice of top performers
    answers_class_1 = [a[0] for a in sorted_data_zeros_first[0:top_number]]
    predictions_class_1 = [a[1] for a in sorted_data_zeros_first[0:top_number]]

    # Class 1 - Slice of top performers
    answers_class_2 = [a[0] for a in sorted_data_ones_first[0:top_number]]
    predictions_class_2 = [a[1] for a in sorted_data_ones_first[0:top_number]]

    # Combined Top and Bottom Confusion Matrix
    tb_answers = answers_class_1 + answers_class_2
    tb_predictions = predictions_class_1 + predictions_class_2

    # Get the TB confusion matrix
    tb_cm = confusion_matrix(tb_answers, tb_predictions, labels=[class_1_label, class_2_label])

    # create the statistics
    stats = calculate_stats(tb_answers, tb_predictions, tb_cm)
    return stats


def generate_markdown_table(corpus_name, search_terms, estimator_parameters, answers, predictions, probabilities,
                            top_number, class_1_label, class_2_label):
    """
    Analyze the top predictions of the model.

    :param corpus_name: String with the name of the dataset we are analyzing
    :param search_terms: List of search terms used to create the context model
    :param estimator_parameters: The hyperparameters used by the estimator
    :param answers: numpy array of ans
    :param predictions: numpy array of predictions
    :param probabilities: numpy array of arrays, each element in the array being a class probability
    :param top_number: the number of top performers to analyze
    :param class_1_label: The label assigned to the second class
    :param class_2_label: The label assigned to the second class
    """

    #
    # Sanitize Input
    #

    if top_number > len(answers):
        raise RuntimeError("Specified number of data points to analyze %s is greater then the number of samples %s" % (
            top_number, len(answers)))

    #
    # Functions
    #

    def create_row_from_list(list_of_values):
        return "| " + " | ".join(list_of_values) + " |"

    def create_table_divider(hl):
        """
        Create the table header and divider

        :param hl: list of header strings
        :return:
        """
        dl = []
        for h in hl:
            dl.append("-" * len(h))
        return dl

    def create_parameter_string_from_list(params):
        """
        Takes a list of parameters and creates table cell content.

        :param params: list of strings
        :return: string with list combined with <br>
        """

        param_string = "<br>".join(params)
        return param_string

    def create_parameter_string(e_params):
        """
        Create the parameter string.

        :param e_params: dictionary of estimator parameters and their corresponding values
        :return:
        """
        param_list = []
        for parameter, value in e_params.items():
            if parameter == 'gamma':
                param_list.append(f"$\gamma$={value}")
            else:
                param_list.append(f"{parameter}={value}")
        param_string = "<br>".join(param_list)
        return param_string

    def create_confusion_matrix_string(cm):
        """
        Create a latex markdown string from the confusion matrix

        :param cm: confusion matrix of form [[TP, FN], [FP, TN]]
        :return: markdown string
        """
        # confusion matrix looks like this [[ 23, 63], [ 40, 260]]
        #                                  [[TP, FN], [FP, TN]]
        begin_matrix = '\\begin{bmatrix}'
        end_matrix = '\end{bmatrix}'
        TP = cm[0][0]
        FN = cm[0][1]
        FP = cm[1][0]
        TN = cm[1][1]
        s = f'${begin_matrix} {TP} & {FN} \\\\ {FP} & {TN} {end_matrix}$'
        return s

    def get_class_distribution(ans, class_1_l, class_2_l):
        total = len(ans)
        class_a = sum([1 for a in ans if a == class_1_l])
        class_b = sum([1 for b in ans if b == class_2_l])
        class_a_percent = class_a / total
        class_b_percent = class_b / total
        lines = [f"N = {total}", f"Class {class_1_l} = {class_a} ({class_a_percent:.2f})",
                 f"Class {class_2_l} = {class_b} ({class_b_percent:.2f})"]
        return "<br>".join(lines)

    def create_stats_string(result_stats):
        max_key_length = max([len(k) for k in result_stats.keys()])
        lines = []
        for key, value in result_stats.items():
            lines.append(f"{key.ljust(max_key_length)} {value}")
        return "<br>".join(lines)

    #
    # Analyze the Data
    #

    # Unpack our probabilities
    prob1 = [p[0] for p in probabilities]
    prob2 = [p[1] for p in probabilities]

    # Create a new data structure so that we can sort
    all_data = list(zip(answers, predictions, prob1, prob2))

    # Sort by probability one, this will correspond to our first class, stance 0, oppose
    sorted_data_zeros_first = sorted(all_data, key=itemgetter(2),
                                     reverse=True)  # Highest prob of getting a 0 is sorted first

    reversed_data = all_data.copy()
    reversed_data.reverse()
    # Sort by probability two, this will correspond to our second class, stance 1, support
    sorted_data_ones_first = sorted(reversed_data, key=itemgetter(3),
                                    reverse=True)  # Highest prob of getting a 1 is at the beginning

    # Class 0 - Slice of top performers
    answers_class_1 = [a[0] for a in sorted_data_zeros_first[0:top_number]]
    predictions_class_1 = [a[1] for a in sorted_data_zeros_first[0:top_number]]

    # Class 1 - Slice of top performers
    answers_class_2 = [a[0] for a in sorted_data_ones_first[0:top_number]]
    predictions_class_2 = [a[1] for a in sorted_data_ones_first[0:top_number]]

    #
    # Create the Table
    #

    header_list = ['Corpus', 'Search Terms', 'Parameters', 'Class Dist', 'Data Set', 'Top/Class a', 'Bottom/Class b', 'Combined', 'TB Stats']
    divider_list = create_table_divider(header_list)
    header_string = create_row_from_list(header_list)
    divider_string = create_row_from_list(divider_list)

    # Full data set confusion matrix
    full_data_set_cm = confusion_matrix(answers, predictions, labels=[class_1_label, class_2_label])
    # Top and Bottom confusion Matrixes
    top_cm = confusion_matrix(answers_class_1, predictions_class_1, labels=[class_1_label, class_2_label])
    bottom_cm = confusion_matrix(answers_class_2, predictions_class_2, labels=[class_1_label, class_2_label])
    # Combined Top and Bottom Confusion Matrix
    tb_answers = answers_class_1 + answers_class_2
    tb_predictions = predictions_class_1 + predictions_class_2
    tb_cm = confusion_matrix(tb_answers, tb_predictions, labels=[class_1_label, class_2_label])
    stats = calculate_stats(tb_answers, tb_predictions, tb_cm)

    # Build the markdown table cell by cell
    cells = [corpus_name,
             create_parameter_string_from_list(search_terms),
             create_parameter_string(estimator_parameters),
             get_class_distribution(answers, class_1_label, class_2_label),
             create_confusion_matrix_string(full_data_set_cm),
             create_confusion_matrix_string(top_cm),
             create_confusion_matrix_string(bottom_cm),
             create_confusion_matrix_string(tb_cm),
             create_stats_string(stats)]

    # Build and return the markdown string
    return header_string + "\n" + divider_string + "\n" + create_row_from_list(cells)