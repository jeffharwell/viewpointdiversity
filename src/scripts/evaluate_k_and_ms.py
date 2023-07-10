import configparser
import pickle
import re
import time

import pandas
import sklearn
from nltk import SnowballStemmer
from sklearn import svm, model_selection

import viewpointdiversitydetection as vdd


#
# Related Keyword Functions
#

def get_pdo(corpus_definition, token_filter, tokenize=False):
    cd = corpus_definition

    # Contact the database
    config = configparser.ConfigParser()
    config.read('config.ini')
    user = config['InternetArgumentCorpus']['username']
    password = config['InternetArgumentCorpus']['password']
    host = config['InternetArgumentCorpus']['host']
    database = 'fourforums'

    # Get the two stances
    rc = vdd.RawCorpusFourForums(cd.topic, database, host, user, password)
    rc.stance_a = cd.stance_a
    rc.stance_b = cd.stance_b
    # rc.print_stats()

    pdo = vdd.ParsedDocumentsFourForums(token_filter, cd.topic, rc.stance_a,
                                    rc.stance_b, database, host, user, password)
    pdo.stance_agreement_cutoff = rc.stance_agreement_cutoff
    label_a = pdo.get_stance_label(rc.stance_a)
    label_b = pdo.get_stance_label(rc.stance_b)
    print(f"{rc.stance_a} => {label_a}, {rc.stance_b} => {label_b}")

    if cd.transform:
        print(f"Setting a pre-parsing transform for {cd.topic}")
        pdo.set_pre_transform(cd.transform)

    pdo.tokenize_only = tokenize

    print("Parsing Documents")
    pdo.process_corpus(print_stats=False)

    return pdo


def indices_and_targets_dataframe(pdo):
    rows = []
    # Determine the record ID
    if hasattr(pdo, 'warc_record_id'):
        # the ID for Commoncrawl is the warc_record_id
        record_id = pdo.warc_record_id
    else:
        # for Fourfourms it is the post_id
        record_id = pdo.post_id

    # Create the dataframe, we save the record ID so that we can compare later if needed
    # to make sure the indicies haven't shifted
    for i, target in enumerate(pdo.target):
        rows.append({'doc_idx': i, 'record_id': record_id[i], 'target': target})
    return pandas.DataFrame(rows)


def get_topic_vectors(pdo_obj, train_df, test_df):
    """
    Create the topic vectors for the test data and the training data.

    :param pdo_obj: A parsed document object with a Spacy parse of the entire corpus
    :param train_df: A training split of the data with document indices in the 'doc_idx' column
    :param test_df: A testing split of the data with document indices in the 'doc_idx' column
    """

    # Create Topic Vectors from training
    topic_vector_obj = vdd.TopicFeatureGenerator()
    topic_vector_obj.workers = 7  # good for an 8 core CPU
    topic_vector_obj.debug = True  # we want the debug output
    topic_vector_obj.min_number_topics = 5  # max coherence seems to land at 4 .. which is a bit low
    topic_vector_obj.create_topic_vectors_from_texts(
        [t for (i, t) in enumerate(pdo_obj.text) if i in train_df['doc_idx'].tolist()])
    print(f"Created a topic vector with {topic_vector_obj.num_topics} topics.")
    print(f"Coherence of the LDA model is {topic_vector_obj.coherence_score}")
    train_topic_vectors = topic_vector_obj.topic_vectors

    # Create the topic feature vectors for the test text using the LDA model we just created
    test_text = [t for (i, t) in enumerate(pdo_obj.text) if i in test_df['doc_idx'].tolist()]
    test_topic_vectors = topic_vector_obj.create_topic_vectors_from_new_text(test_text)

    return train_topic_vectors, test_topic_vectors


def generate_contexts_structure(pdo_obj, search_terms, related_terms, context_size):
    terms_for_extraction = {'search': search_terms, 'related': related_terms}
    ec = vdd.ExtractContexts(pdo_obj, context_size)
    ec.extract_contexts(terms_for_extraction)

    number_of_search_contexts = sum([len(c) for c in ec.get_contexts_by_doc_id_for('search').values()])
    number_of_related_contexts = sum([len(c) for c in ec.get_contexts_by_doc_id_for('related').values()])
    print(f"{number_of_search_contexts} search contexts extracted")
    print(f"{number_of_related_contexts} related contexts extracted")

    return ec

#
# Notebook Functions
#


def tabulate_result_data(run_type, k, mean_match_threshold, use_topics,
                         cd, pdo, ec, related_keywords,
                         best_c, best_gamma, best_class_weight,
                         num_training_documents,
                         data_x, data_x_df,
                         data_y, y_pred, y_pred_prob):
    search_terms = cd.search_terms
    label_a = pdo.get_stance_label(cd.stance_a)
    label_b = pdo.get_stance_label(cd.stance_b)
    overall_balanced_accuracy = sklearn.metrics.balanced_accuracy_score(data_y, y_pred)

    print(f"--- Run Type: {run_type} ---")
    print(
        f"Confusion matrix for the entire test set. Stance '{cd.stance_a}' label '{label_a}' "
        f"is positive. C = {best_c}, gamma = {best_gamma}, class_weight = {best_class_weight}")
    print(sklearn.metrics.confusion_matrix(data_y, y_pred, labels=[label_a, label_b]))
    print("Balanced Accuracy: %.4f" % overall_balanced_accuracy)

    # record our results
    parameters = {'C': best_c, 'gamma': best_gamma, 'class w': best_class_weight,
                  'IAA': pdo.stance_agreement_cutoff, 'word2vec': 'SBERT', 'k': k, 'ms': mean_match_threshold,
                  'topics': str(use_topics),
                  'overall BA': overall_balanced_accuracy}
    corpus_name = 'fourforums' + "<br>" + f"{cd.topic}"
    # Create our statistics from the model for the top 10% most confident predictions
    tbb_stats = vdd.create_run_stats(data_y, y_pred, y_pred_prob,
                                     round(len(data_y) * .1), label_a, label_b)
    # The full markdown table string (for our viewing pleasure)
    table_string = vdd.generate_markdown_table(corpus_name, search_terms,
                                               parameters, data_y, y_pred, y_pred_prob,
                                               round(len(data_y) * .1), label_a, label_b)
    print("TBB Balanced Accuracy: %.4f" % tbb_stats['Bal Acc'])

    # Drop everything back into the dataframe to get some stats on our sentence extraction
    # note that we need to convert the numpy arrays into lists before Pandas will
    # add them correctly.
    result_df = data_x_df.copy()
    result_df['feature_scaled'] = data_x.tolist()
    result_df['predicted'] = y_pred.tolist()
    result_df['probability'] = y_pred_prob.tolist()

    # Grab the extracted sentences and the terms for each document
    result_df[['total_terms', 'terms_matching', 'terms_extracted', 'unique_terms_extracted',
               'total_sentences', 'sentences_extracted', 'percent_terms_extracted',
               'percent_sentences_extracted']] = result_df.apply(
        lambda x: ec.get_sentences_and_terms_by_doc_idx(x['doc_idx']),
        axis='columns', result_type='expand')

    num_corpus_docs = len(pdo.all_docs)
    return {'topic': cd.topic,
            'run_type': run_type,
            'k_value': k,
            'ms_value': mean_match_threshold,
            'topic_vectors': use_topics,
            'num_training_documents': num_training_documents,
            'num_input_documents': len(data_x),
            'num_corpus_documents': num_corpus_docs,
            'target': data_y.copy(),
            'predicted': y_pred.copy(),
            'probability': y_pred_prob.copy(),
            'overall_balanced_accuracy': overall_balanced_accuracy,
            'tbb_balanced_accuracy': tbb_stats['Bal Acc'],
            'tbb_stats': tbb_stats,
            'parameters': parameters.copy(),
            'best_c': best_c,
            'best_gamma': best_gamma,
            'best_class_weight': best_class_weight,
            'IAA': pdo.stance_agreement_cutoff,
            'embedding_type': 'SBERT',
            'overall_confusion_matrix': sklearn.metrics.confusion_matrix(data_y, y_pred, labels=[label_a, label_b]),
            'markdown_string': table_string,
            'num_related_keywords': len(related_keywords),
            'mean_percent_sentences_sampled': result_df.describe().loc['mean']['percent_sentences_extracted'],
            'mean_number_sentences_extracted': result_df.describe().loc['mean']['sentences_extracted'],
            'mean_number_of_terms_matched': result_df.describe().loc['mean']['terms_matching']}


def run_svm_with_params(cd, pdo, related_keywords, ec,
                        best_c, best_gamma, best_class_weight,
                        k, mean_match_threshold, use_topics,
                        train_x_df, test_x_df,
                        train_x, train_y, test_x, test_y):
    # run the SVM with the best parameters
    clf = svm.SVC(probability=True, C=best_c, gamma=best_gamma, class_weight=best_class_weight)
    clf.fit(train_x, train_y)
    y_pred_train = clf.predict(train_x)
    y_pred = clf.predict(test_x)
    prob_test = clf.predict_proba(test_x)
    prob_train = clf.predict_proba(train_x)

    num_training_documents = len(train_x)
    r_train = tabulate_result_data('train', k, mean_match_threshold, use_topics, cd, pdo, ec, related_keywords,
                                   best_c, best_gamma, best_class_weight,
                                   num_training_documents, train_x, train_x_df, train_y, y_pred_train, prob_train)
    r_test = tabulate_result_data('test', k, mean_match_threshold, use_topics, cd, pdo, ec, related_keywords,
                                  best_c, best_gamma, best_class_weight,
                                  num_training_documents, test_x, test_x_df, test_y, y_pred, prob_test)

    return [r_train, r_test]


def add_feature_vectors(doc_idx, feature_vectors):
    """
    Small function that returns the context embedding if it exist, but also sets has_context_embedding
    to a boolean so that rows with no context embedding can be filtered out.
    """
    if doc_idx in feature_vectors:
        return {'context_embedding': feature_vectors[doc_idx], 'has_context_embedding': True}
    else:
        return {'context_embedding': list(), 'has_context_embedding': False}


def verify_indices(pdo, df):
    """
    Make sure that nothing strange happened between our parsed document object and the
    cannonical test/train split. If the record id and document index are different
    between the PDO and the split dataframe then throw an error.

    If the PDO pulled records from its source in a different order then they were pulled
    when the cannonical test/train split was created. Then the record ids and the document indices
    will be off.
    """
    for split_record in df[['doc_idx', 'record_id']].to_dict('records'):
        pdo_record_id = pdo.post_id[split_record['doc_idx']]
        assert pdo_record_id == split_record['record_id']


def run_gridsearch(train_x, train_y):
    """
    Run a grid search on the training data and return the best values of C, gamma, and the class weight

    :param train_x: training data features
    :param train_y: training data targets
    :return:
    """
    tabm = vdd.TopAndBottomMetric(.1)
    skf = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
    params = {'C': [1, 5, 10, 100], 'gamma': [1e-5, 1e-4, 1e-3, 1e-2], 'class_weight': ['balanced']}
    grid = sklearn.model_selection.GridSearchCV(svm.SVC(probability=True), param_grid=params,
                                                scoring=tabm.predictive_value_metric,
                                                cv=skf,
                                                n_jobs=8, error_score='raise')
    grid.fit(train_x, train_y)
    print("Best Params: ", grid.best_params_)

    return grid.best_params_['C'], grid.best_params_['gamma'], grid.best_params_['class_weight']


def create_features(k, ms, ern, pdo, vector_model, cd, extract_context_size):
    """
    Given a value of k and ms, extract the related keywords and generate the features

    :param k:
    :param ms:
    :param ern:
    :param pdo:
    :param vector_model:
    :param cd:
    :param extract_context_size:
    :return: related_keywords, ec, fvt
    """
    # First extract our related terms given the value of k and ms
    related_keywords = ern.get_related_keywords_context_threshold(k, ms)
    print(f"---k={k}, ms={ms}: {len(related_keywords)} returned---")
    print(related_keywords)

    return create_features_from_keyword_list(related_keywords, pdo, vector_model,
                                             cd, extract_context_size)


def create_features_from_keyword_list(related_keywords, pdo, vector_model, cd, extract_context_size):
    """
    Given a value of k and ms, extract the related keywords and generate the features

    :param related_keywords:
    :param pdo:
    :param vector_model:
    :param cd:
    :param extract_context_size:
    :return: related_keywords, ec, fvt
    """
    # Create Feature Vectors for Search and Related Terms
    # We will use the related terms that we learned from the training data to extract contexts
    # from the testing data as well.

    print("Creating Embedding and Sentiment Features from Extracted Contexts")
    fvt = vdd.FeatureVectorsAndTargets(pdo, vector_model, cd.search_terms,
                                       related_keywords, extract_context_size)
    # We need to create our own ExtractContexts structure so that we can grab the
    # sentences later
    ec = generate_contexts_structure(pdo, cd.search_terms, related_keywords, extract_context_size)
    print("Starting Feature Creation")
    start = time.process_time()
    fvt.create_feature_vectors_and_targets()
    end = time.process_time()
    print(f"Created {len(fvt.feature_vectors)} feature vectors in {(end - start) / 60:.2f} minutes.")

    return related_keywords, ec, fvt


def run_experiment_on_corpus(cd, canonical_splits, related_context_size, extract_context_size,
                             tf, stemmer, vector_model):
    """
    Run a full experiment on a corpus

    :param vector_model:
    :param stemmer:
    :param tf:
    :param extract_context_size:
    :param related_context_size:
    :param canonical_splits:
    :param cd: Corpus Definition object
    :return
    """

    results = []
    print("Processing Corpus %s" % cd.topic)

    # Get our processed corpus
    pdo = get_pdo(cd, tf)

    # Pull in our canonical test/train split and put it in the right datastructures
    # for further processing
    c_split = canonical_splits[cd.topic]
    train_x_df = c_split['train_df']
    test_x_df = c_split['test_df']
    verify_indices(pdo, train_x_df)
    verify_indices(pdo, test_x_df)
    print("All Verified")

    print("Size of training set %s" % len(train_x_df))
    print("Size of testing set %s" % len(test_x_df))

    # Get our Topic Vectors
    # Create the LDA model using the training data, then apply it to the test data
    (train_topic_vectors, test_topic_vectors) = get_topic_vectors(pdo, train_x_df, test_x_df)

    print(train_topic_vectors[0])
    print(f"Created {len(train_topic_vectors)} topic vectors for the training data")
    print(f"Created {len(test_topic_vectors)} topic vectors for the testing data")
    assert (len(train_topic_vectors) == len(train_x_df))
    assert (len(test_topic_vectors) == len(test_x_df))

    train_x_df['topic_feature'] = train_topic_vectors
    test_x_df['topic_feature'] = test_topic_vectors

    # Prepare data structures to extract related nouns
    ern = vdd.SelectRelatedKeywords(pdo, train_x_df['doc_idx'].tolist(),
                                    cd.search_terms, stemmer, related_context_size)
    print(ern.coterm_df.describe())

    # Run the loop to evaluate a SVM for every combination of k, ms, and the use of topic vectors
    k_list = [2, 10, 100, 500]
    ms_list = [1, 2]
    use_topic_vectors = [True, False]

    for ms in ms_list:  # two different thresholds for when we stop adding related keywords based on mean terms matched
        for k in k_list:  # different values of k to generate the related keywords

            # Extract our related keywords and create the features
            related_keywords, ec, fvt = create_features(k, ms, ern, pdo, vector_model, cd, extract_context_size)

            # Make a copy so that we don't break anything outside the loop
            train_x_k_df = train_x_df.copy()
            test_x_k_df = test_x_df.copy()

            # Add in our feature vectors
            # Catching the documents that had no context features generated because
            # no contexts were extracted.
            test_x_k_df[['context_embedding', 'has_context_embedding']] = test_x_k_df.apply(
                lambda x: add_feature_vectors(x['doc_idx'], fvt.feature_vectors),
                axis='columns', result_type='expand')
            train_x_k_df[['context_embedding', 'has_context_embedding']] = train_x_k_df.apply(
                lambda x: add_feature_vectors(x['doc_idx'], fvt.feature_vectors),
                axis='columns', result_type='expand')

            # Filter out the rows that don't have a context embedding
            train_x_filtered_df = train_x_k_df[train_x_k_df['has_context_embedding'] == True].copy()
            test_x_filtered_df = test_x_k_df[test_x_k_df['has_context_embedding'] == True].copy()

            for use_topics in use_topic_vectors:  # do we use topic vectors as features or not
                print(f"Using Topics: {use_topics}")

                # Now create the feature vector
                if use_topics:
                    train_x_filtered_df['feature'] = train_x_filtered_df.apply(
                        lambda x: x['context_embedding'] + x['topic_feature'], axis='columns')
                    test_x_filtered_df['feature'] = test_x_filtered_df.apply(lambda x:
                                                                             x['context_embedding']+x['topic_feature'],
                                                                             axis='columns')
                else:
                    train_x_filtered_df['feature'] = train_x_filtered_df['context_embedding']
                    test_x_filtered_df['feature'] = test_x_filtered_df['context_embedding']

                # Extract the features from the dataframe and run them through the scaler
                train_x = train_x_filtered_df['feature'].tolist()
                test_x = test_x_filtered_df['feature'].tolist()

                # don't use the list from the test/train split as we have lost some docs since then
                train_y = train_x_filtered_df['target'].tolist()
                test_y = test_x_filtered_df['target'].tolist()

                #
                # Scale
                #
                scaler = sklearn.preprocessing.StandardScaler()
                scaler.fit(train_x)
                train_x = scaler.transform(train_x)
                test_x = scaler.transform(test_x)

                #
                # Run the SVM Gridsearch
                #
                best_c, best_gamma, best_class_weight = run_gridsearch(train_x, train_y)

                # Now run the SVM
                r = run_svm_with_params(cd, pdo, related_keywords, ec,
                                        best_c, best_gamma, best_class_weight,
                                        k, ms, use_topics,
                                        train_x_filtered_df, test_x_filtered_df,
                                        train_x, train_y, test_x, test_y)
                results += r
    return results


def run_context_size_experiment_on_corpus(cd, canonical_splits, k, ms, use_topic_vectors,
                                          tf, stemmer, vector_model):
    """
    Run a full experiment on a corpus

    :param cd: Corpus Definition object
    :param canonical_splits:
    :param k:
    :param ms:
    :param use_topic_vectors:
    :param tf:
    :param stemmer:
    :param vector_model:
    :return
    """

    results = []
    print("Processing Corpus %s" % cd.topic)

    # Get our processed corpus
    pdo = get_pdo(cd, tf)

    # Pull in our canonical test/train split and put it in the right datastructures
    # for further processing
    c_split = canonical_splits[cd.topic]
    train_x_df = c_split['train_df']
    test_x_df = c_split['test_df']
    verify_indices(pdo, train_x_df)
    verify_indices(pdo, test_x_df)
    print("All Indices Verified")

    print("Size of training set %s" % len(train_x_df))
    print("Size of testing set %s" % len(test_x_df))

    # Get our Topic Vectors if needed
    if use_topic_vectors:
        # Create the LDA model using the training data, then apply it to the test data
        (train_topic_vectors, test_topic_vectors) = get_topic_vectors(pdo, train_x_df, test_x_df)

        print(train_topic_vectors[0])
        print(f"Created {len(train_topic_vectors)} topic vectors for the training data")
        print(f"Created {len(test_topic_vectors)} topic vectors for the testing data")
        assert (len(train_topic_vectors) == len(train_x_df))
        assert (len(test_topic_vectors) == len(test_x_df))

        train_x_df['topic_feature'] = train_topic_vectors
        test_x_df['topic_feature'] = test_topic_vectors

    related_context_sizes = [1, 2, 4, 6, 8]
    extract_context_sizes = [1, 2, 4, 6, 8]

    # create a different list of keywords for each related context size
    for related_context_size in related_context_sizes:
        # Prepare data structures to extract related nouns
        ern = vdd.SelectRelatedKeywords(pdo, train_x_df['doc_idx'].tolist(),
                                        cd.search_terms, stemmer, related_context_size)
        # First extract our related terms given the value of k and ms
        related_keywords = ern.get_related_keywords_context_threshold(k, ms)
        print(f"---k={k}, ms={ms}, related_context_size={related_context_size}: {len(related_keywords)} "
              f"related nouns returned---")
        # print(related_keywords)
        # print(ern.coterm_df.describe())

        # Try a different model for each of our extract context sizes
        for extract_context_size in extract_context_sizes:
            print(f"-- Related Context Size: {related_context_size} Extract Context Size: {extract_context_size} --")
            # Extract our related keywords and create the features
            related_keywords, ec, fvt = create_features_from_keyword_list(related_keywords, pdo,
                                                                          vector_model, cd, extract_context_size)

            # Make a copy so that we don't break anything outside the loop
            train_x_k_df = train_x_df.copy()
            test_x_k_df = test_x_df.copy()

            # Add in our feature vectors
            # Catching the documents that had no context features generated because
            # no contexts were extracted.
            test_x_k_df[['context_embedding', 'has_context_embedding']] = test_x_k_df.apply(
                lambda x: add_feature_vectors(x['doc_idx'], fvt.feature_vectors),
                axis='columns', result_type='expand')
            train_x_k_df[['context_embedding', 'has_context_embedding']] = train_x_k_df.apply(
                lambda x: add_feature_vectors(x['doc_idx'], fvt.feature_vectors),
                axis='columns', result_type='expand')

            # Filter out the rows that don't have a context embedding
            train_x_final = train_x_k_df[train_x_k_df['has_context_embedding'] == True].copy()
            test_x_final = test_x_k_df[test_x_k_df['has_context_embedding'] == True].copy()

            # Now create the feature vector
            if use_topic_vectors:
                print(f"Using Topics: {use_topic_vectors}")
                train_x_final['feature'] = train_x_final.apply(
                    lambda x: x['context_embedding'] + x['topic_feature'], axis='columns')
                test_x_final['feature'] = test_x_final.apply(lambda x: x['context_embedding'] + x['topic_feature'],
                                                             axis='columns')
            else:
                train_x_final['feature'] = train_x_final['context_embedding']
                test_x_final['feature'] = test_x_final['context_embedding']

            # Extract the features from the dataframe and run them through the scaler
            train_x = train_x_final['feature'].tolist()
            test_x = test_x_final['feature'].tolist()

            # don't use the list from the test/train split as we have lost some docs since then
            train_y = train_x_final['target'].tolist()
            test_y = test_x_final['target'].tolist()

            #
            # Scale
            #
            scaler = sklearn.preprocessing.StandardScaler()
            scaler.fit(train_x)
            train_x = scaler.transform(train_x)
            test_x = scaler.transform(test_x)

            #
            # Run the SVM Gridsearch
            #
            best_c, best_gamma, best_class_weight = run_gridsearch(train_x, train_y)

            # Now run the SVM
            r = run_svm_with_params(cd, pdo, related_keywords, ec,
                                    best_c, best_gamma, best_class_weight,
                                    k, ms, use_topic_vectors,
                                    train_x_final, test_x_final,
                                    train_x, train_y, test_x, test_y)
            results += r
    return results


def get_transform(topic):
    if topic == 'abortion':
        # Compile the Regular Expression patterns that we need for the closure
        prolife_pattern_lower = re.compile(r'[p][Rr][Oo][\s-]*[Ll][Ii][Ff][Ee]')
        prochoice_pattern_lower = re.compile(r'[p][Rr][Oo][\s-]*[Cc][Hh][Oo][Ii][Cc][Ee]')
        prolife_pattern_upper = re.compile(r'[P][Rr][Oo][\s-]*[Ll][Ii][Ff][Ee]')
        prochoice_pattern_upper = re.compile(r'[P][Rr][Oo][\s-]*[Cc][Hh][Oo][Ii][Cc][Ee]')

        def transform(text):
            """
            Closure that transforms various forms of pro-life and pro-choice into
            prolife and prochoice so they are not eaten by the tokenizer.
            """
            t1 = prolife_pattern_lower.sub('prolife', text)
            t2 = prochoice_pattern_lower.sub('prochoice', t1)
            t3 = prolife_pattern_upper.sub('Prolife', t2)
            t4 = prochoice_pattern_upper.sub('Prochoice', t3)
            return t4

        return transform
    elif topic in ['existence of God', 'evolution']:
        # We need to limit the length of the texts that we ingest so that we don't crash
        # Spacy when we try to parse it.
        length_limit = 125000
        print(f"Returning transform to cap document length at {length_limit} characters.")

        def transform(text):
            if len(text) > length_limit:
                return text[:length_limit]
            else:
                return text

        return transform
    else:
        return None


def return_corpus_definition_list():
    gun_control_definition = vdd.CorpusDefinition('gun control')
    evolution_definition = vdd.CorpusDefinition('evolution')
    abortion_definition = vdd.CorpusDefinition('abortion')
    existence_of_god_definition = vdd.CorpusDefinition('existence of God')

    gun_control_definition.set_stance('prefers strict gun control',
                                      'opposes strict gun control')
    evolution_definition.set_stance('evolution occurs via purely natural mechanisms',
                                    'evolution involves more than purely natural mechanisms (intelligent design)')
    abortion_definition.set_stance('pro-life', 'pro-choice')
    existence_of_god_definition.set_stance('atheist', 'theist')

    gun_control_definition.set_search_terms(['strict', 'gun', 'control'])
    evolution_definition.set_search_terms(['evolution', 'natural', 'mechanism', 'intelligent', 'design'])
    # should be prolife, and prochoice, not pro-life, and pro-choice, because of the transform rewrites these to avoid
    # them being split into bigrams by the tokenizer!
    abortion_definition.set_search_terms(['abortion', 'prolife', 'prochoice'])
    existence_of_god_definition.set_search_terms(['atheist', 'theist', 'God', 'exist'])

    corpus_definitions = [existence_of_god_definition, abortion_definition, gun_control_definition,
                          evolution_definition]

    # Set the raw text transform functions by topic
    for cd in corpus_definitions:
        cd.set_transform(get_transform(cd.topic))

    return corpus_definitions


if __name__ == '__main__':
    with open('fourforums_cannonical_splits_20230707.pickle', 'rb') as f:
        canonical_splits = pickle.load(f)

    # Initialize our vector model
    vector_model = vdd.SBertFeatureGenerator(False, False)
    # Token Filter
    tf = vdd.TokenFilter()
    # And Stemmer
    stemmer = SnowballStemmer(language='english')
    # Save the results
    context_range_results = []
    # Filename for the pickle that will record our results
    c_pickle_filename = 'context_range_experiment_results_v2_20230709.pickle'

    corpus_definitions = return_corpus_definition_list()

    best_k = 100
    best_ms = 2
    best_topic_vectors = False
    for cd in corpus_definitions:
        r = run_context_size_experiment_on_corpus(cd, canonical_splits,
                                                  best_k, best_ms, best_topic_vectors,
                                                  tf, stemmer, vector_model)
        context_range_results += r
        gc.collect()
        with open(c_pickle_filename, 'wb') as f:
            pickle.dump(context_range_results, f)
        break
