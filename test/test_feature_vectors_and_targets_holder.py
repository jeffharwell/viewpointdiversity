import configparser
import unittest
import gensim.downloader as api

from src.viewpointdiversitydetection import ParsedDocumentsFourForums, FindCharacteristicKeywords
from src.viewpointdiversitydetection.FeatureVectorsAndTargets import *
from src.viewpointdiversitydetection import Holder
from src.viewpointdiversitydetection import TokenFilter


class FeatureVectorsAndTargetsHolderTest(unittest.TestCase):

    def test_holder(self):
        """
        Make sure the feature vector and target Holder class behaves appropriately.
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
        vm_short_name = 'glove-wiki-gigaword-50'
        vector_model = api.load(vm_short_name)
        context_size = 6

        fvt = FeatureVectorsAndTargets(pdo, vector_model, search_terms, related_terms, context_size)
        fvt.create_feature_vectors_and_targets()

        # The holder needs enough information that we can generate a markdown table from
        # the holder and the trained model using the 'generate_markdown_table' function in VDD
        # vdd.generate_markdown_table(corpus_name, search_terms, parameters, test_y, y_pred, prob_test,
        #                             round(len(test_y) * .1), label_a, label_b)
        label_a = pdo.get_stance_label(pdo.stance_a)
        label_b = pdo.get_stance_label(pdo.stance_b)
        holder = Holder(database, pdo.topic_name, fvt.search_terms, pdo.stance_a, pdo.stance_b,
                        label_a, label_b, pdo.stance_agreement_cutoff, vm_short_name)
        holder.populate(fvt)
        self.assertTrue(holder.feature_vectors == fvt.feature_vectors)
        self.assertTrue(holder.feature_vectors_as_components == fvt.feature_vectors_as_components)
        self.assertTrue(holder.targets_for_features == fvt.targets_for_features)
        fvt.targets_for_features.append(2)
        # If they are copies making a change to one should not change the other
        self.assertFalse(holder.targets_for_features == fvt.targets_for_features)


if __name__ == '__main__':
    unittest.main()
