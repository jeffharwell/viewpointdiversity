#!/usr/bin/python


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
import gc
# See https://stackoverflow.com/questions/552744/how-do-i-profile-memory-usage-in-python
import tracemalloc

# Import Viewpoint Diversity Detection Package
import src.viewpointdiversitydetection as vdd

# Get the base stopwords
from nltk.corpus import stopwords

# Word2Vec library
import gensim.downloader as api


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
    # Set the Topic
    #
    # topic = 'gun control'
    topic = 'evolution'

    #
    # Stances
    #
    # stance_a = 'prefers strict gun control'
    # stance_b = 'opposes strict gun control'
    stance_a = 'evolution occurs via purely natural mechanisms'
    stance_b = 'evolution involves more than purely natural mechanisms (intelligent design)'

    #
    # Search Keywords
    #
    # search_terms = ['strict', 'gun', 'control']
    search_terms = ['evolution', 'natural', 'mechanism', 'intelligent', 'design']

    #
    # Load our word2vec model, it is big, do this only once for all subsequent classes
    #

    vector_models = [{'short_name': 'fasttext-wiki', 'api_load': 'fasttext-wiki-news-subwords-300'},
                     {'short_name': 'word2vec-google-news', 'api_load': 'word2vec-google-news-300'},
                     {'short_name': 'glove-gigaword', 'api_load': 'glove-wiki-gigaword-300'}]

    #
    # Load the corpus from the database
    #
    config = configparser.ConfigParser()
    config.read('config.ini')

    user = config['InternetArgumentCorpus']['username']
    password = config['InternetArgumentCorpus']['password']
    host = config['InternetArgumentCorpus']['host']
    database = 'fourforums'
    rc = vdd.RawCorpusFourForums(topic, database, host, user, password)
    rc.stance_a = stance_a
    rc.stance_b = stance_b
    rc.print_stats()

    #
    # Define the tokens we are interested in and then retrieve and pre-process our corpus
    #
    stop_words = set(stopwords.words('english'))
    stop_words = [s for s in stop_words if s not in ['no', 'nor', 'not']]  # I want negations

    def token_filter(spacy_token):
        if not spacy_token.is_space and not spacy_token.is_punct and spacy_token.text.lower() not in stop_words:
            return True
        else:
            return False

    #
    # Parse the corpus
    #

    ## Store and Load this Structure
    ## filename: {topic}_pdo.pickle

    pdo_pickle_name = f"{topic}_pdo.pickle"
    if os.path.exists(pdo_pickle_name):
        infile = open(pdo_pickle_name, 'rb')
        pdo = pickle.load(infile)
        infile.close()
    else:
        pdo = vdd.ParsedDocumentsFourForums(token_filter, topic, rc.stance_a,
                                           rc.stance_b, database, host, user, password)
        pdo.stance_agreement_cutoff = rc.stance_agreement_cutoff
        label_a = pdo.get_stance_label(rc.stance_a)
        label_b = pdo.get_stance_label(rc.stance_b)
        print(f"{rc.stance_a} => {label_a}, {rc.stance_b} => {label_b}")

        pdo.process_corpus()

        #outfile = open(pdo_pickle_name, 'wb')
        #pickle.dump(pdo, outfile)
        #outfile.close()

    #
    # Get the Contexts
    #

    ## Store and Load this Structure as well
    ## should just be able to store the related terms here
    ## filename: {topic}_related_terms.pickle

    pickle_name = f"{topic}_related_terms.pickle"
    if os.path.exists(pickle_name):
        infile = open(pickle_name, 'rb')
        related_terms = pickle.load(infile)
        infile.close()
    else:
        fk = vdd.FindCharacteristicKeywords(pdo)
        related_terms = fk.get_unique_nouns_from_term_context(search_terms)
        print(f"{len(search_terms)} search terms and {len(related_terms)} related terms found for corpus.")
        outfile = open(pickle_name, 'wb')
        pickle.dump(related_terms, outfile)
        outfile.close()
        del fk

    # Get our total coverage
    # This takes some time but the extraction coverage from this last step
    # tells us our overall coverage. Basically how much of the corpus we have
    # access to with the search terms we picked and the immediate context
    # of those terms.
    # fk.get_unique_nouns_from_term_context(related_terms)

    tracemalloc.start()
    for vm in vector_models:
        # load our vector model
        vector_model_short_name = vm['short_name']
        print(f"Loading vector model {vector_model_short_name}:")
        vector_model = api.load(vm['api_load'])

        #
        # Create the Feature Vector from the Contexts
        #
        print("Creating Features")
        context_size = 6
        fvt = vdd.FeatureVectorsAndTargets(pdo, vector_model, search_terms, related_terms, context_size)
        fvt.create_feature_vectors_and_targets()
        print(f"Created {len(fvt.feature_vectors)} feature vectors.")

        print(f"Feature vector size is {len(fvt.feature_vectors[0])}.")

        #
        # Split the Data into Testing and Training
        #

        train_x, test_x, train_y, test_y = model_selection.train_test_split(fvt.feature_vectors,
                                                                            fvt.targets_for_features, test_size=0.2,
                                                                            stratify=fvt.targets_for_features)
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

        pickle_name = f"{topic}_grid.pickle"
        if os.path.exists(pickle_name):
            infile = open(pickle_name, 'rb')
            grid = pickle.load(infile)
            infile.close()
        else:
            tabm = vdd.TopAndBottomMetric(.1)
            skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
            params = {'C': [.1, 1, 5, 10, 100], 'gamma': [1e-5, 1e-4, 1e-3, 1e-2], 'class_weight': ['balanced']}
            grid = GridSearchCV(svm.SVC(probability=True), param_grid=params, scoring=tabm.predictive_value_metric, cv=skf,
                                n_jobs=4, error_score='raise')
            grid.fit(train_x, train_y)
            print("Best Params: ", grid.best_params_)

            outfile = open(pickle_name, 'wb')
            pickle.dump(grid, outfile)
            outfile.close()

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
        # probablities = clf.predict_proba(train_x)
        prob_test = clf.predict_proba(test_x)
        print(
            f"Confusion matrix for the entire test set. Stance '{rc.stance_a}' label '{label_a}' is positive. "
            f"C = {best_c}, gamma = {best_gamma}, class_weight = {best_class_weight}")
        print(confusion_matrix(test_y, y_pred, labels=[label_a, label_b]))
        print("Balanced Accuracy: %.2f" % sklearn.metrics.balanced_accuracy_score(test_y, y_pred))

        #
        # Save the SVM to Disk
        #

        filename = f"./svm_models/SVM-{database}-{topic}-{vector_model_short_name}"
        dump(clf, filename)

        #
        # And Analyze
        #

        # IAA is our iter-annotator agreement cutoff, inter-annotator agreement must be less than or equal to this
        # number in order for the data point to be included.
        parameters = {'C': best_c, 'gamma': best_gamma, 'class w': best_class_weight,
                      'IAA': pdo.stance_agreement_cutoff, 'word2vec': vector_model_short_name}
        corpus_name = database + "<br>" + topic
        table_string = vdd.generate_markdown_table(corpus_name, search_terms, parameters, test_y, y_pred, prob_test,
                                                   round(len(test_y) * .1), label_a, label_b)
        # Markdown String
        print(table_string)

        # Cleanup
        # The models take a lot of memory, clear the variable and do a garbage collection routine
        # before proceeding to process using the next vector model.
        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot)
        print("Cleaning Memory Usage")
        del vector_model
        del fvt
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


if __name__ == '__main__':
    main()
