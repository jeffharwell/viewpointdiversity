import configparser
import unittest
import numpy as np

from viewpointdiversitydetection import SBertFeatureGenerator, FeatureVectorsAndTargets
from viewpointdiversitydetection import TokenFilter, ParsedDocumentsFourForums, FindCharacteristicKeywords


class SBertFeatureGeneratorTest(unittest.TestCase):

    def test_avg_vector_no_divide_by_zero(self):
        """
        When all of the word2vec features are empty we shouldn't hit a divide-by-zero
        RuntimeError if we are not including zeros in the averages.

        :return:
        """
        w2v_obj = SBertFeatureGenerator(False, False)
        zero_vector = np.zeros(w2v_obj.vector_size)
        vectors = [zero_vector.copy(), zero_vector.copy(), zero_vector.copy()]
        avg = w2v_obj._average_vector(vectors)
        self.assertTrue(np.array_equal(avg, zero_vector))

        w2v_obj.include_zeros_in_averages = False

        avg = w2v_obj._average_vector(vectors)
        self.assertTrue(np.array_equal(avg, zero_vector))

    def test_create_document_embeddings(self):
        """
        Does it do anything at all, let's find out.

        :return:
        """
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
        pdo.set_result_limit(10)
        pdo.process_corpus()
        search_terms = ['human', 'responsible', 'climate', 'change']

        fk = FindCharacteristicKeywords(pdo)
        print("\n-- Extracted nouns related to the search terms")
        related_terms = fk.get_unique_nouns_from_term_context(search_terms)

        #
        # Now Create the Feature Vector Object
        #
        # vector_model = api.load('fasttext-wiki-news-subwords-300')
        w2v_obj = SBertFeatureGenerator(False, False)
        context_size = 6

        fvt = FeatureVectorsAndTargets(pdo, w2v_obj, search_terms, related_terms, context_size)
        fvt.create_feature_vectors_and_targets()
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


if __name__ == '__main__':
    unittest.main()