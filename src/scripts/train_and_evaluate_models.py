#!/usr/bin/python
import glob
import linecache
import os

# Our SVM and some associated items
import pickle

from sklearn import svm

from sklearn import metrics
from sklearn import model_selection

import sklearn.metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Analysis
from sklearn.metrics import confusion_matrix

# General Pieces
import configparser
from joblib import dump
# See https://stackoverflow.com/questions/552744/how-do-i-profile-memory-usage-in-python
import tracemalloc

# Import Viewpoint Diversity Detection Package
import viewpointdiversitydetection as vdd


def display_top(snapshot, key_type='lineno', limit=5):
    """
    Display top memory usage.
    From: https://stackoverflow.com/questions/552744/how-do-i-profile-memory-usage-in-python

    :param snapshot: a tracemalloc snapshot
    :param key_type:
    :param limit: number of top memory users to display, defaults to 5
    :return:
    """
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def main():
    #
    # Load the feature holder object from a pickle.
    #
    config = configparser.ConfigParser()
    config.read('config.ini')

    pickle_directory = config['General']['pickle_directory']

    holder_files = glob.glob(f"{pickle_directory}/*holder.pickle")
    print(holder_files)
    table_strings = []
    for filename in holder_files:
        with open(filename, 'rb') as infile:
            fvt_holder = pickle.load(infile)
        print(f"Loading features for {fvt_holder.topic_name} vector model {fvt_holder.vector_model_short_name}")

        #
        # Split the Data into Testing and Training
        #
        train_x, test_x, train_y, test_y = model_selection.train_test_split(fvt_holder.feature_vectors,
                                                                            fvt_holder.targets_for_features, test_size=0.2,
                                                                            stratify=fvt_holder.targets_for_features)
        print("Size of training set %s" % len(train_x))
        print("Size of testing set %s" % len(test_x))

        scaler = StandardScaler()
        scaler.fit(train_x)
        print("Prescaling mean: ", sum(train_x[0]) / len(train_x[0]))
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)
        print("Postscaling mean: ", sum(train_x[0]) / len(train_x[0]))

        # Basic Stats
        total = len(test_y)
        num_a = sum([1 for i in test_y if i == 'a'])
        num_b = sum([1 for i in test_y if i == 'b'])
        print(f"Class a {num_a} of {total}, {num_a / total:.2f}.")
        print(f"Class b {num_b} of {total}, {num_b / total:.2f}.")

        #
        # Find SVM Parameters
        #
        tabm = vdd.TopAndBottomMetric(.1)
        skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
        params = {'C': [.1, 1, 5, 10, 100], 'gamma': [1e-5, 1e-4, 1e-3, 1e-2], 'class_weight': ['balanced']}
        grid = GridSearchCV(svm.SVC(probability=True), param_grid=params, scoring=tabm.predictive_value_metric, cv=skf,
                            n_jobs=4, error_score='raise')
        grid.fit(train_x, train_y)
        print("Best Params: ", grid.best_params_)

        best_c = grid.best_params_['C']
        best_gamma = grid.best_params_['gamma']
        best_class_weight = grid.best_params_['class_weight']

        #
        # Run SVM with the Parameters
        #

        clf = svm.SVC(probability=True, C=best_c, gamma=best_gamma, class_weight=best_class_weight)
        # clf = svm.SVC(probability=True, C=100, gamma=.001, class_weight=class_weight)
        clf.fit(train_x, train_y)
        y_pred = clf.predict(test_x)
        prob_test = clf.predict_proba(test_x)
        print(
            f"Confusion matrix for the entire test set. Stance '{fvt_holder.stance_a}' label '{fvt_holder.label_a}'"
            f"is positive. C = {best_c}, gamma = {best_gamma}, class_weight = {best_class_weight}")
        print(confusion_matrix(test_y, y_pred, labels=[fvt_holder.label_a, fvt_holder.label_b]))
        print("Balanced Accuracy: %.2f" % sklearn.metrics.balanced_accuracy_score(test_y, y_pred))

        #
        # Save the SVM to Disk
        #

        filename = f"./svm_models/SVM-{fvt_holder.database}-{fvt_holder.topic_name}-{fvt_holder.vector_model_short_name}"
        dump(clf, filename)

        #
        # And Analyze
        #

        # IAA is our iter-annotator agreement cutoff, inter-annotator agreement must be less than or equal to this
        # number in order for the data point to be included.
        parameters = {'C': best_c, 'gamma': best_gamma, 'class w': best_class_weight,
                      'IAA': fvt_holder.stance_agreement_cutoff, 'word2vec': fvt_holder.vector_model_short_name}
        corpus_name = fvt_holder.database + "<br>" + fvt_holder.topic_name
        table_string = vdd.generate_markdown_table(corpus_name, fvt_holder.search_terms,
                                                   parameters, test_y, y_pred, prob_test,
                                                   round(len(test_y) * .1), fvt_holder.label_a, fvt_holder.label_b)
        # Markdown String
        print(table_string)
        table_strings.append(table_string)

    print("Resulting Markdown:\n\n")
    for t in table_strings:
        print(t)
        print("\n\n")

    # Cleanup
    # The models take a lot of memory, clear the variable and do a garbage collection routine
    # before proceeding to process using the next vector model.
    """
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)
    print("Cleaning Memory Usage")
    del clf
    del grid
    del train_x
    del test_x
    del train_y
    del test_y
    del scaler
    gc.collect()
    print(gc.garbage)
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)
    """


if __name__ == '__main__':
    main()
