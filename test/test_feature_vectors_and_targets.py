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
        related_terms = fk.get_unique_nouns_from_term_context(search_terms)

        #
        # Now Create the Feature Vector Object
        #
        # vector_model = api.load('fasttext-wiki-news-subwords-300')
        vector_model = api.load('glove-wiki-gigaword-50')
        context_size = 6

        fvt = FeatureVectorsAndTargets(pdo, vector_model, search_terms, related_terms, context_size)
        fvt.create_feature_vectors_and_targets()
        print("First Feature Vector")
        print(fvt.feature_vectors[0])
        print("First Target Class")
        print(fvt.targets_for_features[0])
        print(f"Created {len(fvt.feature_vectors)} feature vectors.")
        self.assertTrue(len(fvt.feature_vectors) > 0)
        self.assertTrue(len(fvt.targets_for_features) > 0)


if __name__ == '__main__':
    unittest.main()