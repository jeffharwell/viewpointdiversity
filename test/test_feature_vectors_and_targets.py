import configparser
import unittest
import gensim.downloader as api

from src.viewpointdiversitydetection import ParsedDocumentsFourForums, FindCharacteristicKeywords
from src.viewpointdiversitydetection.FeatureVectorsAndTargets import *
from nltk.corpus import stopwords

from src.viewpointdiversitydetection import TokenFilter


class FeatureVectorsAndTargetsTest(unittest.TestCase):

    def test_feature_vector_creation(self):
        """
        This is basically our end-to-end test case. Take some data from the database and create feature vectors
        from those documents. The test uses a smaller word2vec model for speed.
        """

        #
        # Retrieve and parse the documents then get the related terms
        #
        tf = TokenFilter()
        config = configparser.ConfigParser()
        config.read('config.ini')

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        pdo = ParsedDocumentsFourForums(tf, 'climate change', 'humans not responsible',
                                        'humans responsible', database, host, user, password)
        pdo.set_result_limit(500)
        pdo.process_corpus()
        search_terms = ['human', 'responsible', 'climate', 'change']

        fk = FindCharacteristicKeywords(pdo)
        print("\n-- Extracted nouns related to the search terms")
        related_terms = fk.get_unique_nouns_from_term_context(search_terms, 'search')

        #
        # Now Create the Feature Vector Object
        #
        # vector_model = api.load('fasttext-wiki-news-subwords-300')
        vector_model = api.load('glove-wiki-gigaword-50')
        context_size = 6

        fvt = FeatureVectorsAndTargets(pdo, vector_model, search_terms, related_terms, context_size)
        fvt.create_feature_vectors_and_targets(feature_type='separate')
        print("First Feature Vector")
        print(fvt.feature_vectors[0])
        print("First Target Class")
        print(fvt.targets_for_features[0])
        print(f"Created {len(fvt.feature_vectors)} feature vectors.")
        self.assertTrue(len(fvt.feature_vectors) > 0)
        self.assertTrue(len(fvt.targets_for_features) > 0)

        # The 'search' and 'related' components together should be equal to the word2vec
        # portion of the full feature array.
        combined = list(fvt.feature_vectors_as_components[0]['search']) + \
                   list(fvt.feature_vectors_as_components[0]['related'])
        combined_array = np.array(combined)
        len_sentiment = len(fvt.feature_vectors_as_components[0]['sentiment'])
        just_w2v = fvt.feature_vectors[0][len_sentiment:]
        self.assertTrue(np.array_equal(combined_array, just_w2v))

    def test_feature_vector_verbose_setting(self):
        """
        Not really a test case, but you should see in the output the result of various
        verbosity settings when creating the feature vectors and targets
        """

        #
        # Retrieve and parse the documents then get the related terms
        #
        tf = TokenFilter()
        config = configparser.ConfigParser()
        config.read('config.ini')

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        pdo = ParsedDocumentsFourForums(tf, 'climate change', 'humans not responsible',
                                        'humans responsible', database, host, user, password)
        pdo.set_result_limit(500)
        pdo.process_corpus()
        search_terms = ['human', 'responsible', 'climate', 'change']

        fk = FindCharacteristicKeywords(pdo)
        print("\n-- Extracted nouns related to the search terms")
        related_terms = fk.get_unique_nouns_from_term_context(search_terms, 'search')

        #
        # Now Create the Feature Vector Object
        #
        # vector_model = api.load('fasttext-wiki-news-subwords-300')
        vector_model = api.load('glove-wiki-gigaword-50')
        context_size = 6

        fvt = FeatureVectorsAndTargets(pdo, vector_model, search_terms, related_terms, context_size)
        print("\n--")
        for v in [0, 1, 2, 3]:
            print(f"Verbose Level = {v}")
            fvt.create_feature_vectors_and_targets(feature_type='separate', verbose_level=v)
            print("")

    def test_feature_type_runtime_error(self):
        """
        Not really a test case, but you should see in the output the result of various
        verbosity settings when creating the feature vectors and targets
        """

        #
        # Retrieve and parse the documents then get the related terms
        #
        tf = TokenFilter()
        config = configparser.ConfigParser()
        config.read('config.ini')

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        pdo = ParsedDocumentsFourForums(tf, 'climate change', 'humans not responsible',
                                        'humans responsible', database, host, user, password)
        pdo.set_result_limit(5)
        pdo.process_corpus()
        search_terms = ['human', 'responsible', 'climate', 'change']

        fk = FindCharacteristicKeywords(pdo)
        # print("\n-- Extracted nouns related to the search terms")
        related_terms = fk.get_unique_nouns_from_term_context(search_terms, 'search')

        #
        # Now Create the Feature Vector Object
        #
        # vector_model = api.load('fasttext-wiki-news-subwords-300')
        vector_model = api.load('glove-wiki-gigaword-50')
        context_size = 6

        fvt = FeatureVectorsAndTargets(pdo, vector_model, search_terms, related_terms, context_size)
        # print("\n--")
        with self.assertRaises(RuntimeError):
            fvt.create_feature_vectors_and_targets('illegal_value', verbose_level=1)

    def test_feature_type_search_only(self):
        """
        Not really a test case, but you should see in the output the result of various
        verbosity settings when creating the feature vectors and targets
        """

        #
        # Retrieve and parse the documents then get the related terms
        #
        tf = TokenFilter()
        config = configparser.ConfigParser()
        config.read('config.ini')

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        pdo = ParsedDocumentsFourForums(tf, 'climate change', 'humans not responsible',
                                        'humans responsible', database, host, user, password)
        pdo.set_result_limit(500)
        pdo.process_corpus()
        search_terms = ['human', 'responsible', 'climate', 'change']

        fk = FindCharacteristicKeywords(pdo)
        # print("\n-- Extracted nouns related to the search terms")
        related_terms = fk.get_unique_nouns_from_term_context(search_terms, 'search')

        #
        # Now Create the Feature Vector Object
        #
        # vector_model = api.load('fasttext-wiki-news-subwords-300')
        vector_model = api.load('glove-wiki-gigaword-50')
        context_size = 6

        fvt = FeatureVectorsAndTargets(pdo, vector_model, search_terms, related_terms, context_size)
        # print("\n--")
        fvt.create_feature_vectors_and_targets('search only', verbose_level=1)
        # print(fvt.feature_vectors[0])
        # print(fvt.feature_vectors_as_components[0])
        search_v = fvt.feature_vectors_as_components[0]['search']
        sentiment_v = fvt.feature_vectors_as_components[0]['sentiment']
        self.assertTrue(fvt.feature_vectors[0] == sentiment_v + list(search_v))

    def test_feature_type_related_only(self):
        """
        Not really a test case, but you should see in the output the result of various
        verbosity settings when creating the feature vectors and targets
        """

        #
        # Retrieve and parse the documents then get the related terms
        #
        tf = TokenFilter()
        config = configparser.ConfigParser()
        config.read('config.ini')

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        pdo = ParsedDocumentsFourForums(tf, 'climate change', 'humans not responsible',
                                        'humans responsible', database, host, user, password)
        pdo.set_result_limit(500)
        pdo.process_corpus()
        search_terms = ['human', 'responsible', 'climate', 'change']

        fk = FindCharacteristicKeywords(pdo)
        # print("\n-- Extracted nouns related to the search terms")
        related_terms = fk.get_unique_nouns_from_term_context(search_terms, 'search')

        #
        # Now Create the Feature Vector Object
        #
        # vector_model = api.load('fasttext-wiki-news-subwords-300')
        vector_model = api.load('glove-wiki-gigaword-50')
        context_size = 6

        fvt = FeatureVectorsAndTargets(pdo, vector_model, search_terms, related_terms, context_size)
        # print("\n--")
        fvt.create_feature_vectors_and_targets('related only', verbose_level=1)
        # print(fvt.feature_vectors[0])
        # print(fvt.feature_vectors_as_components[0])
        related_v = fvt.feature_vectors_as_components[0]['related']
        sentiment_v = fvt.feature_vectors_as_components[0]['sentiment']
        self.assertTrue(fvt.feature_vectors[0] == sentiment_v + list(related_v))

    def test_feature_type_mean(self):
        """
        Not really a test case, but you should see in the output the result of various
        verbosity settings when creating the feature vectors and targets
        """

        #
        # Retrieve and parse the documents then get the related terms
        #
        tf = TokenFilter()
        config = configparser.ConfigParser()
        config.read('config.ini')

        user = config['InternetArgumentCorpus']['username']
        password = config['InternetArgumentCorpus']['password']
        host = config['InternetArgumentCorpus']['host']
        database = 'fourforums'

        pdo = ParsedDocumentsFourForums(tf, 'climate change', 'humans not responsible',
                                        'humans responsible', database, host, user, password)
        pdo.set_result_limit(500)
        pdo.process_corpus()
        search_terms = ['human', 'responsible', 'climate', 'change']

        fk = FindCharacteristicKeywords(pdo)
        print("\n-- Extracted nouns related to the search terms")
        related_terms = fk.get_unique_nouns_from_term_context(search_terms, 'search')

        #
        # Now Create the Feature Vector Object
        #
        # vector_model = api.load('fasttext-wiki-news-subwords-300')
        vector_model = api.load('glove-wiki-gigaword-50')
        context_size = 6

        fvt = FeatureVectorsAndTargets(pdo, vector_model, search_terms, related_terms, context_size)
        print("\n--")
        fvt.create_feature_vectors_and_targets('mean', verbose_level=1)
        print(fvt.feature_vectors[0])
        print(fvt.feature_vectors_as_components[0])
        search_v = fvt.feature_vectors_as_components[0]['search']
        related_v = fvt.feature_vectors_as_components[0]['related']
        sentiment_v = fvt.feature_vectors_as_components[0]['sentiment']
        sentiment_length = len(sentiment_v)
        search_length = len(search_v)
        related_length = len(related_v)

        # figure out how many tests to do
        if len(fvt.feature_vectors_as_components) > 10:
            test_limit = 10
        else:
            test_limit = len(fvt.feature_vectors_as_components)

        self.assertTrue(test_limit > 0)

        # run our tests, verify that the mean is working across approximately 10
        # feature vectors
        for i in range(0, test_limit):
            search_v = fvt.feature_vectors_as_components[i]['search']
            related_v = fvt.feature_vectors_as_components[i]['related']
            sentiment_v = fvt.feature_vectors_as_components[i]['sentiment']

            self.assertTrue(fvt.feature_vectors[i] == sentiment_v + combine_as_average(list(search_v), list(related_v),
                                                                                       include_zeros=False))
            assert (len(fvt.feature_vectors[i]) == sentiment_length + related_length)
            assert (len(fvt.feature_vectors[i]) == sentiment_length + search_length)


if __name__ == '__main__':
    unittest.main()
